package kvruntime

import "fmt"

type NativeResult struct {
	ContextNS      int64            `json:"context_ns"`
	Key            string           `json:"key"`
	Transfer       *TransferResult  `json:"transfer"`
	TransferWallNS int64            `json:"transfer_wall_ns"`
	EnqueueNS      int64            `json:"enqueue_ns"`
	SetupNS        int64            `json:"setup_ns"`
	CleanupNS      int64            `json:"cleanup_ns"`
	ProcessNS      int64            `json:"process_ns"`
	CellsVerified  int              `json:"cells_verified"`
	FinalUsed      map[string]int64 `json:"final_used_bytes"`
	Provenance     string           `json:"provenance"`
}

// RunNativeCase executes allocation, initialization, a real symbolic-memory
// transfer and cleanup. The measured transfer envelope excludes setup/cleanup,
// as in the CUDA fixture, but those operations still consume time and capacity.
func RunNativeCase(profile *Profile, point TransferPoint, sink func(Record)) (*NativeResult, error) {
	spans, gpuBytes, hostBytes, err := NativeSpans(point.Layout, point.Blocks, point.Direction == "h2d")
	if err != nil {
		return nil, err
	}
	alloc, err := profile.Allocation(point)
	if err != nil {
		return nil, err
	}
	e := NewEngine(sink)
	m, err := NewMemory(e, 1024, map[string]int64{"hbm": gpuBytes, "dram": hostBytes})
	if err != nil {
		return nil, err
	}
	r, err := NewRuntime(e, m, profile.QueueDepth)
	if err != nil {
		return nil, err
	}
	if _, err = m.Reserve("gpu", "hbm", "device", gpuBytes); err != nil {
		return nil, err
	}
	if _, err = m.Reserve("host", "dram", point.HostMemory, hostBytes); err != nil {
		return nil, err
	}
	var tail int64
	var contextNS int64
	for _, a := range profile.Allocations {
		if a.NUMA == point.NUMA && a.Blocks == 0 && a.Layout == "all" {
			contextNS = a.StagesNS["cuda_context_init"]
			break
		}
	}
	if contextNS <= 0 {
		return nil, fmt.Errorf("missing measured CUDA context initialization")
	}
	tail = e.MustAdd(Task{Name: "cuda_context_init", Operation: "setup", Resource: "cpu", DurationNS: contextNS})
	phase := "setup"
	stage := func(name, resource string, action func() error) {
		var parents []int64
		if tail != 0 {
			parents = []int64{tail}
		}
		tail = e.MustAdd(Task{Name: name, Operation: phase, Resource: resource, DurationNS: alloc.StagesNS[name], Parents: parents, Finish: action})
	}
	stage("cudaMalloc", "cpu", func() error { return m.Allocate("gpu") })
	stage("host_alloc", "cpu", func() error { return m.Allocate("host") })
	stage("host_first_touch", "cpu", func() error {
		if err := m.Fill("host", 0); err != nil {
			return err
		}
		if point.Direction == "h2d" {
			for _, s := range spans {
				if err := m.SeedRange("host", s.SourceOffset, s.Bytes, 1, s.DestinationOffset); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if point.HostMemory == "registered" {
		stage("host_register", "cpu", func() error { return m.Register("host") })
	}
	stage("gpu_first_touch", "gpu_compute", func() error {
		origin := uint64(1)
		if point.Direction == "h2d" {
			origin = 0
		}
		return m.Fill("gpu", origin)
	})
	stage("stream_events_create", "cpu", nil)
	src, dst := "gpu", "host"
	if point.Direction == "h2d" {
		src, dst = dst, src
	}
	result := &NativeResult{Key: point.Key(), ContextNS: contextNS, Provenance: point.Evidence + " Setup includes one cold CUDA context for this standalone case; source matrix shared a context across cases."}
	tx, err := r.Transfer(TransferSpec{ID: "native_copy", Source: src, Destination: dst, CPU: "cpu", DMA: "dma", Spans: spans, Cost: point.Cost, Parents: []int64{tail}})
	if err != nil {
		return nil, err
	}
	result.Transfer = tx
	verify := e.MustAdd(Task{Name: "verify_contents", Operation: "verify", Resource: "verifier", Parents: []int64{tx.CompletionTask}, Finish: func() error {
		target := m.Buffers[dst]
		expected := make([]Cell, len(target.Cells))
		for i := range expected {
			expected[i] = Cell{0, int64(i) * m.Unit, true}
		}
		for _, s := range spans {
			for j := int64(0); j < s.Bytes; j += m.Unit {
				offset := s.SourceOffset + j
				if point.Direction == "h2d" {
					offset = s.DestinationOffset + j
				}
				expected[(s.DestinationOffset+j)/m.Unit] = Cell{1, offset, true}
			}
		}
		for i, c := range target.Cells {
			if c != expected[i] {
				return fmt.Errorf("data mismatch at byte %d", int64(i)*m.Unit)
			}
		}
		if target.State != "ready" || m.Buffers[src].Pins != 0 {
			return fmt.Errorf("invalid publication/source ownership")
		}
		result.CellsVerified = len(expected)
		return nil
	}})
	tail = verify
	phase = "cleanup"
	stage("stream_events_destroy", "cpu", nil)
	if point.HostMemory == "registered" {
		stage("host_unregister", "cpu", func() error { return m.Unregister("host") })
	}
	stage("host_free", "cpu", func() error { return m.Free("host") })
	stage("cudaFree", "cpu", func() error { return m.Free("gpu") })
	if err = e.Run(); err != nil {
		return nil, err
	}
	if err = m.Check(); err != nil {
		return nil, err
	}
	result.TransferWallNS = tx.CompleteNS - tx.StartNS
	result.EnqueueNS = tx.CPUFinishedNS - tx.StartNS - point.Cost.PlanNS
	result.SetupNS = tx.StartNS
	result.CleanupNS = e.NowNS - tx.CompleteNS
	result.ProcessNS = e.NowNS
	result.FinalUsed = m.Used
	if result.FinalUsed["hbm"] != 0 || result.FinalUsed["dram"] != 0 {
		return nil, fmt.Errorf("native fixture leaked memory")
	}
	return result, nil
}
