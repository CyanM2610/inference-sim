package kvruntime

import "fmt"

type PagedFixtureResult struct {
	Key          string           `json:"key"`
	Step         *PagedStepResult `json:"step"`
	FinalActive  map[string]int64 `json:"final_active_bytes"`
	Retained     map[string]int64 `json:"allocator_retained_bytes"`
	PhysicalPeak map[string]int64 `json:"physical_peak_bytes"`
	Scope        string           `json:"scope"`
}

func RunPagedFixture(p *Profile, point PagedForwardPoint, sink func(Record)) (*PagedFixtureResult, error) {
	const arenaBytes = int64(40 * 917504)
	e := NewEngine(sink)
	capacity := map[string]int64{"hbm": arenaBytes, "hbm_workspace": (40 << 30) - arenaBytes}
	if point.Output != nil {
		capacity["host_output"] = point.Output.HostReservedBytes
	}
	m, err := NewMemory(e, 256, capacity)
	if err != nil {
		return nil, err
	}
	r, err := NewRuntime(e, m, p.QueueDepth)
	if err != nil {
		return nil, err
	}
	if err = m.RetainAllocator("hbm_workspace", point.ReservedBytes-arenaBytes, true); err != nil {
		return nil, err
	}
	if _, err = m.ReserveOpaque("baseline", "hbm_workspace", "opaque_model_input_and_fixed_arena_context", point.BeforeAllocatedBytes-arenaBytes); err != nil {
		return nil, err
	}
	if err = m.Allocate("baseline"); err != nil {
		return nil, err
	}
	for i := 0; i < 40; i++ {
		id := fmt.Sprintf("hbm/slot/%d", i)
		if _, err = m.Reserve(id, "hbm", "paged_arena_slot", 917504); err != nil {
			return nil, err
		}
		if err = m.Allocate(id); err != nil {
			return nil, err
		}
	}
	n := point.Prefix + point.Query
	origins := make([]uint64, n)
	for i := range origins {
		origins[i] = uint64(i + 1)
	}
	var slots []string
	for i := int64(0); i < (n+15)/16; i++ {
		slots = append(slots, fmt.Sprintf("hbm/slot/%d", 17*i%40))
	}
	for page := int64(0); page < (point.Prefix+15)/16; page++ {
		id := slots[page]
		if err = m.BeginWrite(id); err != nil {
			return nil, err
		}
		for tensor := 0; tensor < 56; tensor++ {
			for head := int64(0); head < 4; head++ {
				for token := page * 16; token < min(point.Prefix, (page+1)*16); token++ {
					if err = m.ComputeRange(id, int64(tensor)*16384+head*4096+(token%16)*256, 256, origins[token], int64(tensor*4+int(head))*256); err != nil {
						return nil, err
					}
				}
			}
		}
		if err = m.Publish(id); err != nil {
			return nil, err
		}
	}
	var weights []float64
	for _, c := range p.Compute {
		if c.Length == 256 && c.Mode == "resident" {
			index := 0
			if point.Query == 1 {
				index = 1
			}
			weights = c.ForwardWeights[index]
			break
		}
	}
	key := fmt.Sprintf("paged/p%d/q%d", point.Prefix, point.Query)
	spec := PagedStepSpec{ID: key, Slots: slots, Prefix: point.Prefix, Query: point.Query, TokenOrigins: origins, Weights: weights, Point: point, WorkspacePool: "hbm_workspace"}
	if o := point.Output; o != nil {
		if err = m.RetainAllocator("host_output", o.HostReservedBytes, true); err != nil {
			return nil, err
		}
		if _, err = m.ReserveGranular("host_output", "host_output", "pinned_output_scalar", o.HostStorageBytes, 8); err != nil {
			return nil, err
		}
		if err = m.Allocate("host_output"); err != nil {
			return nil, err
		}
		spec.HostOutputBuffer = "host_output"
		spec.OutputCopyResource = "copy_d2h"
		spec.OutputStreamGroup = "request_stream"
		spec.OutputValue = o.ReferenceToken
		spec.OutputReferenceKind = "calibration_reference"
	}
	step, err := r.PagedStep(spec)
	if err != nil {
		return nil, err
	}
	if err = e.Run(); err != nil {
		return nil, err
	}
	if err = m.Check(); err != nil {
		return nil, err
	}
	for _, b := range m.Buffers {
		if b.Pins != 0 || b.Holds != 0 || b.Writers != 0 {
			return nil, fmt.Errorf("paged fixture leaked references")
		}
	}
	if m.Used["hbm_workspace"] != point.BeforeAllocatedBytes-arenaBytes {
		return nil, fmt.Errorf("workspace did not return to baseline")
	}
	return &PagedFixtureResult{Key: key, Step: step, FinalActive: m.Used, Retained: m.Retained, PhysicalPeak: m.PhysicalPeak, Scope: "Actual warm paged/HF pipeline shape; meaningful KV and undefined tails are distinct. Cached allocator backing is retained, not added to live leases. Activation lifetime is a conservative measured-peak envelope."}, nil
}
