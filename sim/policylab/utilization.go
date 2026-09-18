package policylab

import (
	"fmt"
	"math"
	"reflect"
	"sort"

	"github.com/inference-sim/inference-sim/sim/kv"
)

type UtilizationWindow struct {
	Name       string   `json:"name"`
	StartUS    int64    `json:"start_us"`
	EndUS      int64    `json:"end_us"`
	DurationUS int64    `json:"duration_us"`
	Requests   []string `json:"requests,omitempty"`
}

type UtilizationCapacityPoint struct {
	TimeUS int64            `json:"time_us"`
	Tier   string           `json:"tier"`
	ID     string           `json:"id"`
	Values map[string]int64 `json:"values"`
}

// Each row is a maximal constant interval [StartUS, EndUS). Counts may exceed
// one on parallel resources; busy time uses their union, never the count sum.
type UtilizationActivity struct {
	StartUS        int64  `json:"start_us"`
	EndUS          int64  `json:"end_us"`
	H2DJobs        int64  `json:"h2d_jobs"`
	D2HJobs        int64  `json:"d2h_jobs"`
	GPUJobs        *int64 `json:"gpu_estimated_jobs"`
	BatchEnvelopes int64  `json:"batch_envelopes"`
}

type UtilizationCapacitySummary struct {
	CapacityBlocks int64              `json:"capacity_blocks"`
	BlockBytes     int64              `json:"block_bytes"`
	Mean           map[string]float64 `json:"time_weighted_mean"`
	Peak           map[string]int64   `json:"peak"`
}

type UtilizationWindowSummary struct {
	Window    UtilizationWindow                     `json:"window"`
	Durations map[string]int64                      `json:"durations_us"`
	Fractions map[string]*float64                   `json:"fractions"`
	Capacity  map[string]UtilizationCapacitySummary `json:"capacity"`
}

type UtilizationSummary struct {
	Schema           string                     `json:"schema"`
	TimeUnit         string                     `json:"time_unit"`
	GPUAvailable     bool                       `json:"gpu_estimate_available"`
	GPUCoverage      string                     `json:"gpu_estimate_coverage"`
	CapacityCoverage string                     `json:"capacity_coverage"`
	WindowScope      string                     `json:"window_scope"`
	ProfileWarnings  []ProfileWarning           `json:"profile_warnings,omitempty"`
	Windows          []UtilizationWindowSummary `json:"windows"`
}

type UtilizationData struct {
	Summary  UtilizationSummary
	Capacity []UtilizationCapacityPoint
	Activity []UtilizationActivity
	Phases   []kv.EnginePhaseObservation
}

type utilizationSpan struct {
	start, end int64
	kind       int
}

func utilizationEnd(start, duration int64) (int64, error) {
	if start < 0 || duration < 0 || duration > math.MaxInt64-start {
		return 0, fmt.Errorf("invalid utilization interval")
	}
	return start + duration, nil
}

// BuildUtilization derives separate diagnostics from existing execution data.
// measured selects a wall-clock window, not ownership attribution: shared GPU
// and DMA activity from every request in that window is necessarily included.
func BuildUtilization(r *Result, measured []string) (*UtilizationData, error) {
	if r == nil {
		return nil, fmt.Errorf("missing simulation result")
	}
	u := &UtilizationData{Summary: UtilizationSummary{
		Schema: "hardware_utilization_v1", TimeUnit: "us",
		CapacityCoverage: "event-boundary snapshots; initial empty cache; same-time samples use final state; HBM footprint and allocator commitment are separate",
		WindowScope:      "whole_run starts at zero and includes modeled device tail; measured spans selected actual arrivals to last request completion and includes all system activity in that window",
		ProfileWarnings:  append([]ProfileWarning(nil), r.ProfileWarnings...),
	}, Phases: append([]kv.EnginePhaseObservation(nil), r.EnginePhaseObservations...)}
	var spans []utilizationSpan
	end := int64(0)
	type transfer struct {
		start  int64
		kind   int
		closed bool
	}
	transfers := map[int64]*transfer{}
	compute := map[string]bool{}
	key := func(id string, at int64) string { return fmt.Sprintf("%s/%d", id, at) }
	for _, e := range r.Events {
		if e.Time < 0 {
			return nil, fmt.Errorf("negative event timestamp")
		}
		end = max(end, e.Time)
		switch e.Name {
		case "compute":
			finish, err := utilizationEnd(e.Time, e.Duration)
			if err != nil {
				return nil, err
			}
			spans = append(spans, utilizationSpan{e.Time, finish, 3})
			end = max(end, finish)
			k := key(e.Instance, e.Time)
			if compute[k] {
				return nil, fmt.Errorf("duplicate batch interval %s", k)
			}
			compute[k] = true
		case "transfer_start":
			if transfers[e.Transaction] != nil {
				return nil, fmt.Errorf("duplicate transfer start %d", e.Transaction)
			}
			kind := 1
			if e.Destination == "hbm" {
				kind = 0
			} else if e.Source != "hbm" {
				return nil, fmt.Errorf("unclassified transfer %d", e.Transaction)
			}
			transfers[e.Transaction] = &transfer{start: e.Time, kind: kind}
		case "transfer_end":
			t := transfers[e.Transaction]
			if t == nil || t.closed || e.Time < t.start {
				return nil, fmt.Errorf("invalid transfer completion %d", e.Transaction)
			}
			spans = append(spans, utilizationSpan{t.start, e.Time, t.kind})
			t.closed = true
		}
	}
	for id, t := range transfers {
		if !t.closed {
			return nil, fmt.Errorf("unfinished transfer %d", id)
		}
	}
	for _, at := range r.FinishedUS {
		end = max(end, at)
	}
	u.Summary.GPUAvailable = r.Config.EnginePhases != nil
	u.Summary.GPUCoverage = "existing engine phase model: [max(forward_ready, previous_gpu_ready), gpu_ready); execution envelope estimate, not SM occupancy, measured kernels, or FLOP utilization"
	seen := map[string]bool{}
	for _, phase := range u.Phases {
		end = max(end, phase.AdoptUS)
		if !phase.HasCompute {
			continue
		}
		k := key(phase.Instance, phase.StartUS)
		if seen[k] || !compute[k] {
			return nil, fmt.Errorf("phase lacks unique batch interval %s", k)
		}
		seen[k] = true
		end = max(end, phase.GPUReadyUS)
		if phase.GPUStartEstimateUS < phase.StartUS || phase.GPUReadyUS < phase.GPUStartEstimateUS {
			u.Summary.GPUAvailable = false
		} else {
			spans = append(spans, utilizationSpan{phase.GPUStartEstimateUS, phase.GPUReadyUS, 2})
		}
	}
	if len(seen) != len(compute) {
		u.Summary.GPUAvailable = false
	}
	if !u.Summary.GPUAvailable {
		u.Summary.GPUCoverage = "unavailable: missing engine phase observations/model or GPU ready precedes the forward-ready estimate; batch envelope is not substituted for GPU busy time"
	}
	var err error
	u.Activity, err = utilizationActivity(spans, end, u.Summary.GPUAvailable)
	if err != nil {
		return nil, err
	}
	u.Capacity, err = utilizationCapacity(r, end)
	if err != nil {
		return nil, err
	}
	windows := []UtilizationWindow{{Name: "whole_run", EndUS: end, DurationUS: end}}
	if len(measured) > 0 {
		arrival := map[string]int64{}
		for _, q := range r.Config.Requests {
			arrival[q.ID] = q.At
		}
		for id, at := range r.ArrivalUS {
			arrival[id] = at
		}
		w := UtilizationWindow{Name: "measured", StartUS: math.MaxInt64, Requests: append([]string(nil), measured...)}
		selected := map[string]bool{}
		for _, id := range measured {
			at, ok := arrival[id]
			done, finished := r.FinishedUS[id]
			if !ok || !finished || selected[id] || at < 0 || done < at || done > end {
				return nil, fmt.Errorf("invalid utilization measurement request %q", id)
			}
			selected[id] = true
			w.StartUS = min(w.StartUS, at)
			w.EndUS = max(w.EndUS, done)
		}
		w.DurationUS = w.EndUS - w.StartUS
		windows = append(windows, w)
	}
	for _, w := range windows {
		u.Summary.Windows = append(u.Summary.Windows, u.summarize(w))
	}
	return u, nil
}

func utilizationActivity(spans []utilizationSpan, end int64, gpu bool) ([]UtilizationActivity, error) {
	deltas := map[int64][4]int64{0: {}, end: {}}
	for _, s := range spans {
		if s.kind == 2 && !gpu || s.start == s.end {
			continue
		}
		if s.start < 0 || s.end < s.start || s.end > end {
			return nil, fmt.Errorf("invalid activity span")
		}
		d := deltas[s.start]
		d[s.kind]++
		deltas[s.start] = d
		d = deltas[s.end]
		d[s.kind]--
		deltas[s.end] = d
	}
	times := make([]int64, 0, len(deltas))
	for t := range deltas {
		times = append(times, t)
	}
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	var rows []UtilizationActivity
	var active [4]int64
	for i, t := range times {
		if i > 0 && t > times[i-1] {
			row := UtilizationActivity{StartUS: times[i-1], EndUS: t, H2DJobs: active[0], D2HJobs: active[1], BatchEnvelopes: active[3]}
			if gpu {
				v := active[2]
				row.GPUJobs = &v
			}
			rows = append(rows, row)
		}
		for j, delta := range deltas[t] {
			active[j] += delta
			if active[j] < 0 {
				return nil, fmt.Errorf("negative activity count")
			}
		}
	}
	if active != [4]int64{} {
		return nil, fmt.Errorf("activity did not drain")
	}
	return rows, nil
}

func utilizationCapacity(r *Result, end int64) ([]UtilizationCapacityPoint, error) {
	groups := map[string][]UtilizationCapacityPoint{}
	add := func(t int64, tier, id string, m map[string]int64) error {
		v := map[string]int64{"capacity_blocks": m["capacity"], "ready_blocks": m["ready"]}
		if tier == "hbm" {
			v["resident_blocks"], v["reserved_blocks"] = m["resident"], m["pending_hbm_targets"]
			v["footprint_blocks"] = m["resident"] + m["pending_hbm_targets"]
			v["allocator_committed_blocks"], v["allocator_free_blocks"] = m["active_or_pinned"], m["free"]
			for target, source := range map[string]string{"store_pinned_blocks": "store_pinned_blocks", "held_restore_blocks": "held_restores", "request_refs": "request_refs", "total_refs": "total_refs", "decode_reserved_blocks": "decode_reserved_blocks"} {
				v[target] = m[source]
			}
		} else {
			v["reserved_blocks"], v["footprint_blocks"], v["read_pins"] = m["reserved"], m["ready"]+m["reserved"], m["read_pins"]
		}
		if v["capacity_blocks"] <= 0 || v["footprint_blocks"] > v["capacity_blocks"] || v["ready_blocks"] > v["capacity_blocks"] || r.BlockBytes <= 0 {
			return fmt.Errorf("invalid capacity snapshot %s/%s", tier, id)
		}
		for _, n := range v {
			if n < 0 {
				return fmt.Errorf("negative capacity counter")
			}
		}
		for _, name := range []string{"capacity", "footprint", "ready", "reserved"} {
			n := v[name+"_blocks"]
			if n > math.MaxInt64/r.BlockBytes {
				return fmt.Errorf("capacity bytes overflow")
			}
			v[name+"_bytes"] = n * r.BlockBytes
		}
		k := tier + "/" + id
		rows := groups[k]
		point := UtilizationCapacityPoint{TimeUS: t, Tier: tier, ID: id, Values: v}
		if len(rows) > 0 && t < rows[len(rows)-1].TimeUS {
			return fmt.Errorf("unordered capacity snapshots")
		}
		if len(rows) > 0 && t == rows[len(rows)-1].TimeUS {
			rows[len(rows)-1] = point
		} else if len(rows) == 0 || !reflect.DeepEqual(rows[len(rows)-1].Values, v) {
			rows = append(rows, point)
		}
		groups[k] = rows
		return nil
	}
	for i, c := range r.Config.Instances {
		if err := add(0, "hbm", fmt.Sprintf("instance_%d", i), map[string]int64{"capacity": c.HBMBlocks, "free": c.HBMBlocks}); err != nil {
			return nil, err
		}
	}
	for _, c := range r.Config.Pools {
		if err := add(0, "pool", c.ID, map[string]int64{"capacity": c.CapacityBlocks}); err != nil {
			return nil, err
		}
	}
	for _, e := range r.Events {
		var err error
		if e.Name == "hbm_capacity" {
			err = add(e.Time, "hbm", e.Instance, e.Counters)
		}
		if e.Name == "l2_capacity" {
			err = add(e.Time, "pool", e.Destination, e.Counters)
		}
		if err != nil {
			return nil, err
		}
	}
	var rows []UtilizationCapacityPoint
	for _, g := range groups {
		last := g[len(g)-1]
		if last.TimeUS < end {
			last.TimeUS = end
			g = append(g, last)
		}
		rows = append(rows, g...)
	}
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.TimeUS != b.TimeUS {
			return a.TimeUS < b.TimeUS
		}
		if a.Tier != b.Tier {
			return a.Tier < b.Tier
		}
		return a.ID < b.ID
	})
	return rows, nil
}

func utilizationRatio(n, d int64) *float64 {
	if d == 0 {
		return nil
	}
	v := float64(n) / float64(d)
	return &v
}

func (u *UtilizationData) summarize(w UtilizationWindow) UtilizationWindowSummary {
	s := UtilizationWindowSummary{Window: w, Durations: map[string]int64{}, Fractions: map[string]*float64{}, Capacity: map[string]UtilizationCapacitySummary{}}
	for _, key := range []string{"h2d_busy", "d2h_busy", "transfer_any_busy", "duplex_busy", "batch_envelope"} {
		s.Durations[key] = 0
	}
	if u.Summary.GPUAvailable {
		for _, key := range []string{"gpu_estimated_busy", "gpu_and_h2d", "gpu_and_d2h", "gpu_and_transfer", "gpu_and_duplex"} {
			s.Durations[key] = 0
		}
	}
	for _, r := range u.Activity {
		d := max(int64(0), min(r.EndUS, w.EndUS)-max(r.StartUS, w.StartUS))
		if d == 0 {
			continue
		}
		h, l, g := r.H2DJobs > 0, r.D2HJobs > 0, r.GPUJobs != nil && *r.GPUJobs > 0
		conditions := map[string]bool{"h2d_busy": h, "d2h_busy": l, "transfer_any_busy": h || l, "duplex_busy": h && l, "batch_envelope": r.BatchEnvelopes > 0,
			"gpu_estimated_busy": g, "gpu_and_h2d": g && h, "gpu_and_d2h": g && l, "gpu_and_transfer": g && (h || l), "gpu_and_duplex": g && h && l}
		for key, active := range conditions {
			if active {
				s.Durations[key] += d
			}
		}
	}
	for key, n := range s.Durations {
		s.Fractions[key+"_fraction"] = utilizationRatio(n, w.DurationUS)
	}
	for _, key := range []string{"gpu_estimated_busy_fraction", "gpu_and_transfer_fraction", "compute_overlap_fraction", "transfer_overlap_fraction"} {
		if _, ok := s.Fractions[key]; !ok {
			s.Fractions[key] = nil
		}
	}
	if u.Summary.GPUAvailable {
		s.Fractions["compute_overlap_fraction"] = utilizationRatio(s.Durations["gpu_and_transfer"], s.Durations["gpu_estimated_busy"])
		s.Fractions["transfer_overlap_fraction"] = utilizationRatio(s.Durations["gpu_and_transfer"], s.Durations["transfer_any_busy"])
	}
	previous := map[string]UtilizationCapacityPoint{}
	for _, p := range u.Capacity {
		k := p.Tier + "/" + p.ID
		if old, ok := previous[k]; ok {
			d := max(int64(0), min(p.TimeUS, w.EndUS)-max(old.TimeUS, w.StartUS))
			if d > 0 {
				c := s.Capacity[k]
				if c.Mean == nil {
					c = UtilizationCapacitySummary{CapacityBlocks: old.Values["capacity_blocks"], BlockBytes: old.Values["capacity_bytes"] / old.Values["capacity_blocks"], Mean: map[string]float64{}, Peak: map[string]int64{}}
				}
				for name, value := range old.Values {
					c.Mean[name] += float64(value) * float64(d) / float64(w.DurationUS)
					c.Peak[name] = max(c.Peak[name], value)
				}
				s.Capacity[k] = c
			}
		}
		previous[k] = p
	}
	for k, c := range s.Capacity {
		for _, name := range []string{"footprint", "ready", "reserved", "allocator_committed"} {
			if v, ok := c.Mean[name+"_blocks"]; ok {
				c.Mean[name+"_fraction"] = v / float64(c.CapacityBlocks)
			}
		}
		s.Capacity[k] = c
	}
	return s
}
