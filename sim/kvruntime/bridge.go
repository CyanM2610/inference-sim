package kvruntime

import "fmt"

type BridgeResult struct {
	StagesNS       map[string]int64  `json:"stages_ns"`
	TransactionNS  int64             `json:"transaction_ns"`
	Transfers      []*TransferResult `json:"transfers"`
	PeakExtraBytes int64             `json:"peak_extra_bytes"`
	MappingKernels int               `json:"mapping_kernels"`
	Evidence       string            `json:"evidence"`
}

// PagedBridge executes the measured HF->paged->DRAM->paged->HF transaction.
// current contains the 56 live HF tensors and is replaced with restored tensors.
// Reordering kernels, actual copy spans and temporary storage are explicit.
func (r *Runtime) PagedBridge(tokens int64, current []string, p *BridgePoint) (*BridgeResult, error) {
	if p == nil || len(current) != 56 {
		return nil, fmt.Errorf("bridge requires 56 tensor endpoints and calibration")
	}
	scatter, err := HFScatterSpans(tokens)
	if err != nil {
		return nil, err
	}
	e, m := r.Engine, r.Memory
	start := e.NowNS
	before := m.Used["hbm"]
	result := &BridgeResult{StagesNS: map[string]int64{}, Evidence: p.Evidence}
	blocks := tokens / 16
	tensorBytes := tokens * 1024
	arena, host, restored := make([]string, 56), make([]string, 56), make([]string, 56)
	for i := range arena {
		arena[i] = fmt.Sprintf("bridge/arena0/%d", i)
		host[i] = fmt.Sprintf("bridge/host/%d", i)
		restored[i] = fmt.Sprintf("bridge/hf/%d", i)
	}
	alloc := func(id, pool, kind string, bytes int64) error {
		if _, err := m.Reserve(id, pool, kind, bytes); err != nil {
			return err
		}
		return m.Allocate(id)
	}
	phase := func(name string, action func() error) error {
		c, ok := p.Stages[name]
		if !ok || c.WallNS < 0 {
			return fmt.Errorf("missing bridge stage %s", name)
		}
		begin := e.NowNS
		e.MustAdd(Task{Name: name, Operation: "bridge", Resource: "cpu", DurationNS: c.WallNS, Finish: action})
		if err := e.Run(); err != nil {
			return err
		}
		result.StagesNS[name] = e.NowNS - begin
		return nil
	}
	if err = phase("arena_and_host_allocate", func() error {
		for i := range arena {
			if err := alloc(arena[i], "hbm", "paged_arena", 2*tensorBytes); err != nil {
				return err
			}
			if err := alloc(host[i], "dram", "pinned_torch_allocator", tensorBytes); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	mapPhase := func(name string, gather bool) error {
		c, ok := p.Stages[name]
		if !ok || c.SubmitNS <= 0 || c.ServiceNS <= 0 {
			return fmt.Errorf("invalid mapping calibration")
		}
		begin := e.NowNS
		var cpuTail, gpuTail int64
		kernel := func(k MappingKernel) error {
			if cpuTail != 0 {
				k.Parents = []int64{cpuTail}
			}
			k.StreamParent = gpuTail
			k.SubmitNS = c.SubmitNS
			k.ServiceNS = c.ServiceNS
			var err error
			cpuTail, gpuTail, err = r.MapKernel(name, "cpu", "mapping_gpu", k)
			result.MappingKernels++
			return err
		}
		for i := range arena {
			if !gather {
				if err := kernel(MappingKernel{Name: "index_copy_scatter", Source: current[i], Destination: arena[i], Spans: scatter}); err != nil {
					return err
				}
				continue
			}
			temporary := fmt.Sprintf("bridge/index_select_temp/%d", i)
			destination := restored[i]
			var selected, transpose []Span
			for b := int64(0); b < blocks; b++ {
				slot := 2 * ((37 * b) % blocks)
				selected = append(selected, Span{slot * 16384, b * 16384, 16384})
				for h := int64(0); h < 4; h++ {
					transpose = append(transpose, Span{b*16384 + h*4096, (h*tokens + b*16) * 256, 4096})
				}
			}
			if err := kernel(MappingKernel{Name: "index_select_pages", Source: arena[i], Destination: temporary, Spans: selected, Prepare: func() error { return alloc(temporary, "hbm", "index_select_temporary", tensorBytes) }}); err != nil {
				return err
			}
			if err := kernel(MappingKernel{Name: "transpose_contiguous", Source: temporary, Destination: destination, Spans: transpose, Prepare: func() error { return alloc(destination, "hbm", "hf_restored_cache", tensorBytes) }, Complete: func() error { return m.Free(temporary) }}); err != nil {
				return err
			}
		}
		e.MustAdd(Task{Name: "mapping_completion_observe", Operation: name, Resource: "cpu", DurationNS: c.ObserveNS, Parents: []int64{cpuTail, gpuTail}})
		if err := e.Run(); err != nil {
			return err
		}
		result.StagesNS[name] = e.NowNS - begin
		return nil
	}
	if err = mapPhase("hf_to_paged_scatter", false); err != nil {
		return nil, err
	}
	transfer := func(direction string) error {
		cost, ok := p.Transfers[direction]
		if !ok {
			return fmt.Errorf("missing bridge transfer calibration")
		}
		var spans []Span
		var endpoints []CopyEndpoints
		for i := range arena {
			for b := int64(0); b < blocks; b++ {
				slot := 2 * ((37 * b) % blocks)
				src, dst := arena[i], host[i]
				so, do := slot*16384, b*16384
				if direction == "h2d" {
					src, dst = dst, src
					so, do = do, so
				}
				spans = append(spans, Span{so, do, 16384})
				endpoints = append(endpoints, CopyEndpoints{src, dst})
			}
		}
		name := "paged_" + direction
		begin := e.NowNS
		tx, err := r.Transfer(TransferSpec{ID: "bridge/" + name, CPU: "cpu", DMA: "pcie_copy", Spans: spans, Endpoints: endpoints, Cost: cost})
		if err != nil {
			return err
		}
		result.Transfers = append(result.Transfers, tx)
		if err := e.Run(); err != nil {
			return err
		}
		result.StagesNS[name] = e.NowNS - begin
		return nil
	}
	if err = transfer("d2h"); err != nil {
		return nil, err
	}
	if err = phase("source_and_arena_release", func() error {
		for i := range arena {
			if err := m.Free(current[i]); err != nil {
				return err
			}
			if err := m.Free(arena[i]); err != nil {
				return err
			}
			arena[i] = fmt.Sprintf("bridge/arena1/%d", i)
		}
		if m.Buffers["prefill_storage_overhead"] != nil {
			return m.Free("prefill_storage_overhead")
		}
		return nil
	}); err != nil {
		return nil, err
	}
	if err = phase("restore_arena_allocate", func() error {
		for _, id := range arena {
			if err := alloc(id, "hbm", "paged_restore_arena", 2*tensorBytes); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	if err = transfer("h2d"); err != nil {
		return nil, err
	}
	if err = mapPhase("paged_to_hf_gather", true); err != nil {
		return nil, err
	}
	if err = phase("publish_and_release_bridge", func() error {
		for i := range arena {
			current[i] = restored[i]
			if err := m.Free(arena[i]); err != nil {
				return err
			}
		}
		e.Emit(Record{Name: "host_lease_return", Reason: "torch_pinned_storage_retained_in_allocator"})
		return nil
	}); err != nil {
		return nil, err
	}
	e.MustAdd(Task{Name: "bridge_python_and_measurement_bookkeeping", Operation: "bridge", Resource: "cpu", DurationNS: p.ResidualNS})
	if err = e.Run(); err != nil {
		return nil, err
	}
	result.StagesNS["python_and_measurement_bookkeeping"] = p.ResidualNS
	result.TransactionNS = e.NowNS - start
	result.PeakExtraBytes = m.Peak["hbm"] - before
	return result, m.Check()
}
