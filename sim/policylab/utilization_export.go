package policylab

import (
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
)

var utilizationCapacityColumns = []string{
	"capacity_blocks", "footprint_blocks", "resident_blocks", "ready_blocks", "reserved_blocks",
	"allocator_committed_blocks", "allocator_free_blocks", "store_pinned_blocks", "held_restore_blocks",
	"decode_reserved_blocks", "read_pins", "request_refs", "total_refs",
	"capacity_bytes", "footprint_bytes", "ready_bytes", "reserved_bytes",
}

func utilizationJSON(path string, value any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	e := json.NewEncoder(f)
	e.SetIndent("", "  ")
	err = e.Encode(value)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func utilizationCSV(path string, header []string, write func(*csv.Writer) error) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	err = w.Write(header)
	if err == nil {
		err = write(w)
	}
	w.Flush()
	if err == nil {
		err = w.Error()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

// WriteUtilization keeps utilization curves and ratios separate from request
// latency/cache summaries and the original event/Perfetto exports.
func WriteUtilization(dir string, r *Result, measured []string) error {
	u, err := BuildUtilization(r, measured)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	if err = utilizationJSON(filepath.Join(dir, "summary.json"), u.Summary); err != nil {
		return err
	}
	n := func(v int64) string { return strconv.FormatInt(v, 10) }
	err = utilizationCSV(filepath.Join(dir, "kv-capacity.csv"), append([]string{"time_us", "tier", "id"}, utilizationCapacityColumns...), func(w *csv.Writer) error {
		for _, p := range u.Capacity {
			row := []string{n(p.TimeUS), p.Tier, p.ID}
			for _, key := range utilizationCapacityColumns {
				value := ""
				if v, ok := p.Values[key]; ok {
					value = n(v)
				}
				row = append(row, value)
			}
			if err := w.Write(row); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	bit := func(b bool) string {
		if b {
			return "1"
		}
		return "0"
	}
	err = utilizationCSV(filepath.Join(dir, "activity.csv"), []string{"start_us", "end_us", "h2d_busy", "d2h_busy", "transfer_any_busy", "duplex_busy", "gpu_estimated_busy", "compute_transfer_overlap", "batch_envelope_busy", "h2d_jobs", "d2h_jobs", "gpu_estimated_jobs"}, func(w *csv.Writer) error {
		for _, a := range u.Activity {
			gpu, overlap, jobs := "", "", ""
			if a.GPUJobs != nil {
				gpu = bit(*a.GPUJobs > 0)
				overlap = bit(*a.GPUJobs > 0 && (a.H2DJobs > 0 || a.D2HJobs > 0))
				jobs = n(*a.GPUJobs)
			}
			row := []string{n(a.StartUS), n(a.EndUS), bit(a.H2DJobs > 0), bit(a.D2HJobs > 0), bit(a.H2DJobs > 0 || a.D2HJobs > 0), bit(a.H2DJobs > 0 && a.D2HJobs > 0), gpu, overlap, bit(a.BatchEnvelopes > 0), n(a.H2DJobs), n(a.D2HJobs), jobs}
			if err := w.Write(row); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(dir, "gpu-phases.jsonl"))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	for _, phase := range u.Phases {
		if err = enc.Encode(phase); err != nil {
			f.Close()
			return err
		}
	}
	if err = f.Close(); err != nil {
		return err
	}
	return utilizationJSON(filepath.Join(dir, "timeline.json"), u.perfetto())
}

func (u *UtilizationData) perfetto() map[string]any {
	var events []map[string]any
	lanes := map[string]int{}
	lane := func(name string) int {
		if id, ok := lanes[name]; ok {
			return id
		}
		id := len(lanes) + 1
		lanes[name] = id
		events = append(events, map[string]any{"ph": "M", "pid": 1, "tid": id, "name": "thread_name", "args": map[string]string{"name": name}})
		return id
	}
	counter := func(name string, t int64, args map[string]int64) {
		id := lane(name)
		events = append(events, map[string]any{"ph": "C", "pid": 1, "tid": id, "name": name, "ts": t, "args": args})
	}
	for _, p := range u.Capacity {
		args := map[string]int64{}
		for _, key := range []string{"capacity_blocks", "footprint_blocks", "ready_blocks", "reserved_blocks", "allocator_committed_blocks", "allocator_free_blocks", "decode_reserved_blocks"} {
			if v, ok := p.Values[key]; ok {
				args[key] = v
			}
		}
		counter(p.Tier+"/"+p.ID+" KV capacity", p.TimeUS, args)
	}
	bit := func(b bool) int64 {
		if b {
			return 1
		}
		return 0
	}
	for _, a := range u.Activity {
		counter("H2D busy", a.StartUS, map[string]int64{"busy": bit(a.H2DJobs > 0)})
		counter("D2H busy", a.StartUS, map[string]int64{"busy": bit(a.D2HJobs > 0)})
		counter("bidirectional transfer overlap", a.StartUS, map[string]int64{"busy": bit(a.H2DJobs > 0 && a.D2HJobs > 0)})
		counter("batch envelope (host and waits included)", a.StartUS, map[string]int64{"busy": bit(a.BatchEnvelopes > 0)})
		if a.GPUJobs != nil {
			counter("GPU execution estimate", a.StartUS, map[string]int64{"busy": bit(*a.GPUJobs > 0)})
			counter("GPU estimate and transfer overlap", a.StartUS, map[string]int64{"busy": bit(*a.GPUJobs > 0 && (a.H2DJobs > 0 || a.D2HJobs > 0))})
		}
	}
	// Close the step curves explicitly at the modeled device-drain boundary.
	end := u.Summary.Windows[0].Window.EndUS
	for _, name := range []string{"H2D busy", "D2H busy", "bidirectional transfer overlap", "batch envelope (host and waits included)"} {
		counter(name, end, map[string]int64{"busy": 0})
	}
	if u.Summary.GPUAvailable {
		counter("GPU execution estimate", end, map[string]int64{"busy": 0})
		counter("GPU estimate and transfer overlap", end, map[string]int64{"busy": 0})
		for _, p := range u.Phases {
			if !p.HasCompute || p.GPUReadyUS == p.GPUStartEstimateUS {
				continue
			}
			id := lane(p.Instance + " GPU phase estimate")
			events = append(events, map[string]any{"ph": "X", "pid": 1, "tid": id, "name": "modeled GPU execution envelope", "ts": p.GPUStartEstimateUS, "dur": p.GPUReadyUS - p.GPUStartEstimateUS, "args": p})
		}
	}
	for _, s := range u.Summary.Windows {
		id := lane("measurement windows")
		events = append(events, map[string]any{"ph": "X", "pid": 1, "tid": id, "name": s.Window.Name, "ts": s.Window.StartUS, "dur": s.Window.DurationUS, "args": s.Window})
	}
	return map[string]any{"traceEvents": events, "displayTimeUnit": "ms"}
}
