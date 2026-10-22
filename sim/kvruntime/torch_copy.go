package kvruntime

import "fmt"

type TorchCopyResult struct {
	Key           string           `json:"key"`
	Transfer      *TransferResult  `json:"transfer"`
	CellsVerified int64            `json:"cells_verified"`
	FinalActive   map[string]int64 `json:"final_active_bytes"`
	PhysicalPeak  map[string]int64 `json:"physical_peak_bytes"`
	Scope         string           `json:"scope"`
}

// RunTorchCopyCase reproduces the native helper's actual tensor and slot
// addresses in a warm model context. The model is opaque during the isolated
// transaction. Setup measurements remain in the profile, outside this window.
func RunTorchCopyCase(p *Profile, point TransferPoint, sink func(Record)) (*TorchCopyResult, error) {
	c := p.TorchCopies
	if c == nil {
		return nil, fmt.Errorf("missing live Torch transfer profile")
	}
	e := NewEngine(sink)
	m, err := NewMemory(e, 256, map[string]int64{"hbm": 40 * 917504, "dram": 64 * 917504, "hbm_workspace": (40 << 30) - 40*917504})
	if err != nil {
		return nil, err
	}
	r, err := NewRuntime(e, m, p.QueueDepth)
	if err != nil {
		return nil, err
	}
	if err = m.RetainAllocator("hbm_workspace", c.ReservedBytes-40*917504, true); err != nil {
		return nil, err
	}
	if _, err = m.ReserveOpaque("model_context", "hbm_workspace", "opaque_warm_model_context", c.BaselineAllocatedBytes-40*917504); err != nil {
		return nil, err
	}
	if err = m.Allocate("model_context"); err != nil {
		return nil, err
	}
	var spans []Span
	var endpoints []CopyEndpoints
	for tensor := 0; tensor < 56; tensor++ {
		gpu, host := fmt.Sprintf("gpu/%d", tensor), fmt.Sprintf("host/%d", tensor)
		for _, v := range []struct {
			id, pool, kind string
			slots          int64
		}{{gpu, "hbm", "device_arena", 40}, {host, "dram", "pinned_arena", 64}} {
			if _, err = m.Reserve(v.id, v.pool, v.kind, v.slots*16384); err != nil {
				return nil, err
			}
			if err = m.Allocate(v.id); err != nil {
				return nil, err
			}
			origin := uint64(0)
			if (v.pool == "hbm") == (point.Direction == "d2h") {
				origin = uint64(tensor + 1)
			}
			if err = m.Fill(v.id, origin); err != nil {
				return nil, err
			}
		}
		for b := int64(0); b < point.Blocks; b++ {
			slot := 17 * b % 40
			spans = append(spans, Span{slot * 16384, slot * 16384, 16384})
			ep := CopyEndpoints{gpu, host}
			if point.Direction == "h2d" {
				ep = CopyEndpoints{host, gpu}
			}
			endpoints = append(endpoints, ep)
		}
	}
	tx, err := r.Transfer(TransferSpec{ID: "torch_native_copy", CPU: "copy_thread", DMA: "pcie_copy", Spans: spans, Endpoints: endpoints, Cost: point.Cost})
	if err != nil {
		return nil, err
	}
	if err = e.Run(); err != nil {
		return nil, err
	}
	var verified int64
	used := map[int64]bool{}
	for b := int64(0); b < point.Blocks; b++ {
		used[17*b%40] = true
	}
	for tensor := 0; tensor < 56; tensor++ {
		id := fmt.Sprintf("host/%d", tensor)
		if point.Direction == "h2d" {
			id = fmt.Sprintf("gpu/%d", tensor)
		}
		for i, cell := range m.Buffers[id].Cells {
			origin := uint64(0)
			if used[int64(i)*256/16384] {
				origin = uint64(tensor + 1)
			}
			if cell != (Cell{origin, int64(i) * 256, true}) {
				return nil, fmt.Errorf("native Torch copy content/hole mismatch %s offset=%d", id, i*256)
			}
			verified++
		}
	}
	if err = m.Check(); err != nil {
		return nil, err
	}
	for _, b := range m.Buffers {
		if b.Pins != 0 || b.Holds != 0 || b.Writers != 0 {
			return nil, fmt.Errorf("in-process copy leaked references")
		}
	}
	return &TorchCopyResult{Key: point.Key(), Transfer: tx, CellsVerified: verified, FinalActive: m.Used, PhysicalPeak: m.PhysicalPeak, Scope: "Measured warm 7B Torch process; 56 separate tensors, GPU 40 slots, pinned host 64 slots, permutation 17*b mod 40. Model/allocator backing retained. Isolated transfer window excludes recorded startup and does not simulate concurrent forward."}, nil
}
