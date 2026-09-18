package policylab

import (
	"encoding/csv"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kv"
)

func utilizationFixture() *Result {
	r := &Result{BlockBytes: 100, Config: Config{EnginePhases: &EnginePhaseCostConfig{}, Instances: []InstanceConfig{{HBMBlocks: 10}}, Pools: []kv.PeerPoolConfig{{ID: "dram", CapacityBlocks: 20}},
		Requests: []RequestConfig{{ID: "warm", At: 0}, {ID: "measured", At: 6}}}, FinishedUS: map[string]int64{"warm": 20, "measured": 14}}
	transfer := func(id, start, end int64, load bool) {
		source, dest := "hbm", "dram"
		if load {
			source, dest = dest, source
		}
		r.Events = append(r.Events,
			kv.PeerRecord{Transaction: id, Name: "transfer_start", Time: start, Source: source, Destination: dest, Duration: end - start},
			kv.PeerRecord{Transaction: id, Name: "transfer_end", Time: end, Source: source, Destination: dest, Duration: end - start})
	}
	transfer(1, 0, 8, true)
	transfer(2, 4, 10, true)
	transfer(3, 6, 12, false)
	r.Events = append(r.Events, kv.PeerRecord{Name: "compute", Instance: "instance_0", Time: 2, Duration: 16},
		kv.PeerRecord{Name: "hbm_capacity", Instance: "instance_0", Time: 2, Counters: map[string]int64{"capacity": 10, "resident": 4, "pending_hbm_targets": 1, "active_or_pinned": 2, "free": 8, "ready": 3}},
		kv.PeerRecord{Name: "hbm_capacity", Instance: "instance_0", Time: 2, Counters: map[string]int64{"capacity": 10, "resident": 5, "pending_hbm_targets": 1, "active_or_pinned": 2, "free": 8, "ready": 3}},
		kv.PeerRecord{Name: "hbm_capacity", Instance: "instance_0", Time: 10, Counters: map[string]int64{"capacity": 10, "resident": 6, "active_or_pinned": 0, "free": 10, "ready": 6}},
		kv.PeerRecord{Name: "l2_capacity", Destination: "dram", Time: 3, Counters: map[string]int64{"capacity": 20, "ready": 2, "reserved": 2}},
		kv.PeerRecord{Name: "l2_capacity", Destination: "dram", Time: 9, Counters: map[string]int64{"capacity": 20, "ready": 4}})
	sort.SliceStable(r.Events, func(i, j int) bool { return r.Events[i].Time < r.Events[j].Time })
	r.EnginePhaseObservations = []kv.EnginePhaseObservation{{Instance: "instance_0", StartUS: 2, HasCompute: true, ForwardReadyUS: 5, GPUStartEstimateUS: 5, GPUReadyUS: 15, AdoptUS: 18},
		{Instance: "instance_0", StartUS: 18, HasCompute: false, GPUReadyUS: 15, AdoptUS: 20}}
	return r
}

func TestUtilizationUnionOverlapAndMeasuredWindow(t *testing.T) {
	r := utilizationFixture()
	before, _ := json.Marshal(r)
	u, err := BuildUtilization(r, []string{"measured"})
	if err != nil {
		t.Fatal(err)
	}
	if !u.Summary.GPUAvailable {
		t.Fatal(u.Summary.GPUCoverage)
	}
	w := u.Summary.Windows[0]
	want := map[string]int64{"h2d_busy": 10, "d2h_busy": 6, "transfer_any_busy": 12, "duplex_busy": 4, "batch_envelope": 16, "gpu_estimated_busy": 10, "gpu_and_h2d": 5, "gpu_and_d2h": 6, "gpu_and_transfer": 7, "gpu_and_duplex": 4}
	if !reflect.DeepEqual(w.Durations, want) {
		t.Fatalf("union/intersection != hand calculation: %v", w.Durations)
	}
	if *w.Fractions["compute_overlap_fraction"] != .7 || *w.Fractions["transfer_overlap_fraction"] != 7.0/12 || *w.Fractions["gpu_estimated_busy_fraction"] != .5 {
		t.Fatal(w.Fractions)
	}
	m := u.Summary.Windows[1]
	if m.Window.StartUS != 6 || m.Window.EndUS != 14 || m.Durations["gpu_and_transfer"] != 6 || *m.Fractions["compute_overlap_fraction"] != .75 || *m.Fractions["transfer_overlap_fraction"] != 1 {
		t.Fatal(m)
	}
	if after, _ := json.Marshal(r); string(before) != string(after) {
		t.Fatal("utilization mutated source result")
	}
	for _, row := range u.Activity {
		if row.EndUS <= row.StartUS {
			t.Fatal("zero/negative activity interval")
		}
	}
}

func TestUtilizationCapacityTimeWeightingAndCachedFreeSpace(t *testing.T) {
	u, err := BuildUtilization(utilizationFixture(), nil)
	if err != nil {
		t.Fatal(err)
	}
	h := u.Summary.Windows[0].Capacity["hbm/instance_0"]
	if math.Abs(h.Mean["footprint_blocks"]-5.4) > 1e-12 || math.Abs(h.Mean["allocator_committed_blocks"]-.8) > 1e-12 || h.Peak["footprint_blocks"] != 6 {
		t.Fatal(h)
	}
	pool := u.Summary.Windows[0].Capacity["pool/dram"]
	if math.Abs(pool.Mean["footprint_blocks"]-3.4) > 1e-12 {
		t.Fatal(pool)
	}
	if _, ok := pool.Mean["allocator_committed_blocks"]; ok {
		t.Fatal("pool read pins fabricated allocator ownership")
	}
	for _, p := range u.Capacity {
		if p.Tier == "hbm" && p.TimeUS == 10 && (p.Values["allocator_free_blocks"] != 10 || p.Values["footprint_blocks"] != 6) {
			t.Fatal("free cached blocks treated as empty", p)
		}
	}
}

func TestUtilizationUnavailableGPUIsNotZeroOrBatchEnvelope(t *testing.T) {
	for _, invalid := range []string{"no_model", "missing_observation", "ready_before_forward"} {
		r := utilizationFixture()
		switch invalid {
		case "no_model":
			r.Config.EnginePhases = nil
		case "missing_observation":
			r.EnginePhaseObservations = nil
		case "ready_before_forward":
			r.EnginePhaseObservations[0].GPUReadyUS = 3
		}
		u, err := BuildUtilization(r, nil)
		if err != nil {
			t.Fatal(err)
		}
		w := u.Summary.Windows[0]
		if u.Summary.GPUAvailable || w.Fractions["gpu_estimated_busy_fraction"] != nil || w.Durations["batch_envelope"] != 16 {
			t.Fatal(invalid, w)
		}
		if _, ok := w.Durations["gpu_estimated_busy"]; ok {
			t.Fatal("unavailable GPU exported as zero")
		}
		for _, a := range u.Activity {
			if a.GPUJobs != nil {
				t.Fatal("unavailable GPU curve fabricated")
			}
		}
	}
}

func TestUtilizationRejectsMissingTransferAndInvalidCohort(t *testing.T) {
	r := utilizationFixture()
	r.Events = append(r.Events, kv.PeerRecord{Name: "transfer_start", Transaction: 99, Time: 20, Destination: "hbm"})
	if _, err := BuildUtilization(r, nil); err == nil {
		t.Fatal("missing DMA completion accepted")
	}
	for _, ids := range [][]string{{"unknown"}, {"measured", "measured"}} {
		if _, err := BuildUtilization(utilizationFixture(), ids); err == nil {
			t.Fatal("invalid cohort accepted")
		}
	}
	r = utilizationFixture()
	r.ArrivalUS = map[string]int64{"measured": 9}
	u, err := BuildUtilization(r, []string{"measured"})
	if err != nil || u.Summary.Windows[1].Window.StartUS != 9 {
		t.Fatal("closed-loop arrival ignored", err)
	}
}

func TestUtilizationTouchingIntervalsAndZeroDenominators(t *testing.T) {
	rows, err := utilizationActivity([]utilizationSpan{{0, 5, 0}, {5, 10, 1}, {3, 3, 2}}, 10, true)
	if err != nil {
		t.Fatal(err)
	}
	u := &UtilizationData{Summary: UtilizationSummary{GPUAvailable: true}, Activity: rows}
	w := u.summarize(UtilizationWindow{EndUS: 10, DurationUS: 10})
	if w.Durations["duplex_busy"] != 0 || w.Fractions["compute_overlap_fraction"] != nil || *w.Fractions["transfer_overlap_fraction"] != 0 {
		t.Fatal(w)
	}
	empty := u.summarize(UtilizationWindow{})
	for _, v := range empty.Fractions {
		if v != nil {
			t.Fatal("zero denominator must be null")
		}
	}
}

func TestUtilizationExportsIndependentCurves(t *testing.T) {
	dir := t.TempDir()
	if err := WriteUtilization(dir, utilizationFixture(), []string{"measured"}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"summary.json", "kv-capacity.csv", "activity.csv", "gpu-phases.jsonl", "timeline.json"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || len(data) == 0 {
			t.Fatal(name, err)
		}
	}
	f, err := os.Open(filepath.Join(dir, "activity.csv"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if rows[0][6] != "gpu_estimated_busy" || rows[1][0] != "0" {
		t.Fatal(rows[:2])
	}
	data, _ := os.ReadFile(filepath.Join(dir, "timeline.json"))
	var trace struct {
		Events []map[string]any `json:"traceEvents"`
	}
	if err := json.Unmarshal(data, &trace); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, e := range trace.Events {
		if e["name"] == "modeled GPU execution envelope" {
			found = true
			if e["ts"] != float64(5) || e["dur"] != float64(10) {
				t.Fatal(e)
			}
		}
	}
	if !found {
		t.Fatal("missing GPU estimate slice")
	}
}

func TestUtilizationActualEnginePhaseBoundaries(t *testing.T) {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	c.TraceBatchShapes = true
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	u, err := BuildUtilization(r, []string{c.Requests[len(c.Requests)-1].ID})
	if err != nil {
		t.Fatal(err)
	}
	if !u.Summary.GPUAvailable || len(u.Phases) == 0 {
		t.Fatal("phase observations missing")
	}
	for _, p := range u.Phases {
		if p.StartUS > p.PlanningEndUS || p.PlanningEndUS > p.PreForwardUS || p.PreForwardUS > p.ForwardReadyUS || p.ForwardReadyUS > p.PostForwardUS || p.PostForwardUS > p.LoadSubmitEndUS || p.LoadSubmitEndUS > p.PollUS || p.PollUS > p.AdoptUS {
			t.Fatal("phase ordering", p)
		}
		if p.HasCompute && (p.GPUStartEstimateUS != max(p.ForwardReadyUS, p.PreviousGPUReadyUS) || p.GPUReadyUS-p.ForwardReadyUS != 95) {
			t.Fatal("estimate did not use exact existing phase times", p)
		}
	}
	if u.Summary.Windows[0].Durations["gpu_estimated_busy"] == 0 {
		t.Fatal("no GPU estimate")
	}
}
