package kvruntime

import (
	"encoding/json"
	"fmt"
)

const modelHeads = int64(4)
const modelHeadTokenBytes = int64(256)
const modelTensors = 56

type ModelResult struct {
	Bridge              *BridgeResult     `json:"bridge,omitempty"`
	Key                 string            `json:"key"`
	SetupNS             int64             `json:"setup_ns"`
	InputNS             int64             `json:"input_ns"`
	PrefillNS           int64             `json:"prefill_ns"`
	TransactionNS       int64             `json:"transaction_ns"`
	DecodeNS            []int64           `json:"decode_ns"`
	CleanupNS           int64             `json:"cleanup_ns"`
	ResumeFirstNS       int64             `json:"resume_first_ns"`
	TransactionDecodeNS int64             `json:"transaction_decode_ns"`
	ProcessNS           int64             `json:"process_ns"`
	Transfers           []*TransferResult `json:"transfers"`
	KVCellsVerified     int64             `json:"kv_cells_verified"`
	FinalKVTokens       int64             `json:"final_kv_tokens"`
	PeakUsed            map[string]int64  `json:"peak_used_bytes"`
	FinalUsed           map[string]int64  `json:"final_used_bytes"`
	ReferenceTokens     []int64           `json:"reference_tokens"`
	Scope               string            `json:"scope"`
}

// RunModelCase executes symbolic HF DynamicCache growth and native save/restore.
// Only the measured 7B batch-one shapes are accepted. Neural numerical values
// and allocator workspace internals remain opaque, with explicit accounting.
func RunModelCase(p *Profile, c ComputePoint, sink func(Record)) (*ModelResult, error) {
	if c.Length != 256 && c.Length != 1024 && c.Length != 2048 {
		return nil, fmt.Errorf("unmeasured model length")
	}
	if c.Mode != "resident" && c.Mode != "native_new" && c.Mode != "native_pool" && c.Mode != "paged_bridge" {
		return nil, fmt.Errorf("unknown model mode")
	}
	if len(c.DecodeNS) != 8 || len(c.ReferenceTokens) != 8 || len(c.ForwardWeights) != 9 || c.BaselineAllocatedBytes <= 0 {
		return nil, fmt.Errorf("model structural calibration required")
	}
	e := NewEngine(sink)
	m, err := NewMemory(e, 256, map[string]int64{"hbm": 40 << 30, "dram": 8 << 30})
	if err != nil {
		return nil, err
	}
	runtime, err := NewRuntime(e, m, p.QueueDepth)
	if err != nil {
		return nil, err
	}
	result := &ModelResult{Key: fmt.Sprintf("model/%d/%s", c.Length, c.Mode), ReferenceTokens: c.ReferenceTokens, Scope: c.StructuralEvidence + " Symbolic KV lineage, not numerical token prediction. Non-KV baseline is measured opaque allocation. Transient activation/allocator reserve footprint and process teardown latency are not calibrated. Setup is a single-shape session; hardware probe pools all three lengths."}
	stage := func(name, resource string, ns int64, action func() error) error {
		e.MustAdd(Task{Name: name, Operation: result.Key, Resource: resource, DurationNS: ns, Finish: action})
		return e.Run()
	}
	alloc := func(id, pool, kind string, bytes int64, opaque bool) error {
		var err error
		if opaque {
			_, err = m.ReserveOpaque(id, pool, kind, bytes)
		} else {
			_, err = m.Reserve(id, pool, kind, bytes)
		}
		if err != nil {
			return err
		}
		return m.Allocate(id)
	}
	if err = stage("cuda_context_init", "cpu", c.ContextInitNS, nil); err != nil {
		return nil, err
	}
	if err = stage("model_load_and_device_transfer", "loader_envelope", c.ModelLoadNS, func() error {
		return alloc("baseline", "hbm", "opaque_model_and_persistent_tensors", c.BaselineAllocatedBytes, true)
	}); err != nil {
		return nil, err
	}
	tensorBytes := c.Length * modelHeads * modelHeadTokenBytes
	hosts := make([]string, modelTensors)
	current := make([]string, modelTensors)
	for i := range hosts {
		hosts[i] = fmt.Sprintf("host/%d", i)
	}
	hostAllocate := func() error {
		for _, id := range hosts {
			if m.Buffers[id] == nil {
				if err := alloc(id, "dram", "pinned_torch_allocator", tensorBytes, false); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if c.Mode == "native_pool" {
		if err = stage("host_pool_allocate_and_touch", "cpu", c.PoolSetupNS, func() error {
			if err := hostAllocate(); err != nil {
				return err
			}
			for _, id := range hosts {
				if err := m.Fill(id, 0); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return nil, err
		}
	}
	result.SetupNS = e.NowNS
	if err = stage("input_ids_h2d_and_sync", "input_envelope", c.InputCopyNS, nil); err != nil {
		return nil, err
	}
	result.InputNS = c.InputCopyNS
	// A forward emits layer boundaries at measured structural fractions. Each
	// layer creates two new KV tensors while the preceding version is pinned.
	forward := func(step int, ns int64) error {
		weights := c.ForwardWeights[step]
		if len(weights) != 30 {
			return fmt.Errorf("expected 30 structural intervals")
		}
		var tail int64
		var allocatedNS int64
		for segment, w := range weights {
			if w < 0 {
				return fmt.Errorf("negative forward fraction")
			}
			duration := int64(w * float64(ns))
			if segment == 29 {
				duration = ns - allocatedNS
			}
			allocatedNS += duration
			name := "forward_prelude"
			if segment > 0 && segment < 29 {
				name = fmt.Sprintf("layer_%02d_forward_envelope", segment-1)
			}
			if segment == 29 {
				name = "logits_argmax_and_sync"
			}
			var parents []int64
			if tail != 0 {
				parents = []int64{tail}
			}
			layer := segment - 1
			var ids, old []string
			tail = e.MustAdd(Task{Name: name, Operation: fmt.Sprintf("forward/%d", step), Resource: "forward_envelope", DurationNS: duration, Parents: parents, Start: func() error {
				if layer < 0 || layer >= 28 {
					return nil
				}
				for j := 0; j < 2; j++ {
					i := 2*layer + j
					id := fmt.Sprintf("kv/%d/%d", step, i)
					ids = append(ids, id)
					old = append(old, current[i])
					n := c.Length + int64(step)
					if err := alloc(id, "hbm", "hf_dynamic_cache", n*modelHeads*modelHeadTokenBytes, false); err != nil {
						return err
					}
					if err := m.BeginWrite(id); err != nil {
						return err
					}
					if step > 0 {
						if err := m.Pin(current[i]); err != nil {
							return err
						}
					}
				}
				return nil
			}, Finish: func() error {
				if layer < 0 || layer >= 28 {
					return nil
				}
				for j, id := range ids {
					i := 2*layer + j
					n := c.Length + int64(step)
					if step == 0 {
						if err := m.ComputeRange(id, 0, n*modelHeads*modelHeadTokenBytes, uint64(i+1), 0); err != nil {
							return err
						}
					} else {
						for h := int64(0); h < modelHeads; h++ {
							if err := m.Copy(old[j], id, h*(n-1)*modelHeadTokenBytes, h*n*modelHeadTokenBytes, (n-1)*modelHeadTokenBytes); err != nil {
								return err
							}
							if err := m.ComputeRange(id, (h*n+n-1)*modelHeadTokenBytes, modelHeadTokenBytes, uint64(1000+step*modelTensors+i), h*modelHeadTokenBytes); err != nil {
								return err
							}
						}
					}
					if err := m.Publish(id); err != nil {
						return err
					}
					if step > 0 {
						if err := m.Unpin(old[j]); err != nil {
							return err
						}
						if err := m.Free(old[j]); err != nil {
							return err
						}
					}
					current[i] = id
				}
				return nil
			}})
		}
		if err := e.Run(); err != nil {
			return err
		}
		token := c.PrefillReferenceToken
		if step > 0 {
			token = c.ReferenceTokens[step-1]
		}
		e.Emit(Record{Name: "reference_token_available", Operation: fmt.Sprintf("forward/%d", step), Origin: uint64(token), Reason: "workload_reference_not_prediction"})
		return nil
	}
	if err = forward(0, c.PrefillNS); err != nil {
		return nil, err
	}
	result.PrefillNS = c.PrefillNS
	if c.Mode == "paged_bridge" {
		if c.Bridge == nil {
			return nil, fmt.Errorf("missing paged bridge calibration")
		}
		if c.Bridge.PrefillStorageBytes > 0 {
			if err = alloc("prefill_storage_overhead", "hbm", "opaque_prefill_retained_storage", c.Bridge.PrefillStorageBytes, true); err != nil {
				return nil, err
			}
		}
		result.Scope += " Paged bridge transaction uses its own measured Python scatter/copy/gather profile. Source-associated non-KV storage is explicitly accounted and released from measured allocation deltas. Prefill/setup/cleanup outside the bridge still use the earlier model profile."
	}
	scalar := func(key string) int64 { var n int64; _ = json.Unmarshal(c.StagesNS[key], &n); return n }
	phase := func(key string, action func() error) error { return stage(key, "cpu", scalar(key), action) }
	transactionStart := e.NowNS
	if c.Mode == "paged_bridge" {
		result.Bridge, err = runtime.PagedBridge(c.Length, current, c.Bridge)
		if err != nil {
			return nil, err
		}
		result.Transfers = append(result.Transfers, result.Bridge.Transfers...)
	} else if c.Mode != "resident" {
		if err = phase("host_allocate", hostAllocate); err != nil {
			return nil, err
		}
		copyAll := func(direction string) error {
			key := "store"
			if direction == "h2d" {
				key = "restore"
			}
			var times map[string]int64
			if err := json.Unmarshal(c.StagesNS[key], &times); err != nil {
				return err
			}
			// PyTorch 56-copy envelopes, distinct from the raw CUDA microbenchmark.
			submit := max(1, times["enqueue"]/modelTensors)
			dma := max(1, (times["event_span"]-submit)/modelTensors)
			predicted := max(submit*modelTensors, submit+dma*modelTensors)
			cost := TransferCost{SubmitNS: submit, DMANS: dma, ObserveNS: max(0, times["wall"]-predicted)}
			var spans []Span
			var endpoints []CopyEndpoints
			for i := 0; i < modelTensors; i++ {
				src, dst := current[i], hosts[i]
				if direction == "h2d" {
					src, dst = dst, src
				}
				spans = append(spans, Span{0, 0, tensorBytes})
				endpoints = append(endpoints, CopyEndpoints{src, dst})
			}
			tx, err := runtime.Transfer(TransferSpec{ID: key, CPU: "cpu", DMA: "dma", Spans: spans, Endpoints: endpoints, Cost: cost})
			if err != nil {
				return err
			}
			result.Transfers = append(result.Transfers, tx)
			return e.Run()
		}
		if err = phase("store_plan", nil); err != nil {
			return nil, err
		}
		if err = copyAll("d2h"); err != nil {
			return nil, err
		}
		if err = phase("source_release", func() error {
			for _, id := range current {
				if err := m.Free(id); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return nil, err
		}
		if err = phase("hbm_allocate", func() error {
			for i := range current {
				current[i] = fmt.Sprintf("restored/%d", i)
				if err := alloc(current[i], "hbm", "hf_restored_cache", tensorBytes, false); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return nil, err
		}
		if err = phase("restore_plan", nil); err != nil {
			return nil, err
		}
		if err = copyAll("h2d"); err != nil {
			return nil, err
		}
		if err = phase("publish", nil); err != nil {
			return nil, err
		}
		if err = phase("host_release", func() error {
			e.Emit(Record{Name: "host_lease_return", Reason: "pinned_allocator_storage_retained_until_session_end"})
			return nil
		}); err != nil {
			return nil, err
		}
	}
	if err = stage("transaction_python_residual", "cpu", c.TransactionResidualNS, nil); err != nil {
		return nil, err
	}
	result.TransactionNS = e.NowNS - transactionStart
	for step, ns := range c.DecodeNS {
		start := e.NowNS
		if err = forward(step+1, ns); err != nil {
			return nil, err
		}
		result.DecodeNS = append(result.DecodeNS, e.NowNS-start)
	}
	if err = stage("decode_loop_python_residual", "cpu", c.LoopResidualNS, nil); err != nil {
		return nil, err
	}
	result.TransactionDecodeNS = e.NowNS - transactionStart
	result.ResumeFirstNS = result.TransactionNS + result.DecodeNS[0]
	n := c.Length + 8
	result.FinalKVTokens = n
	// Independent coordinate reference: prefix offsets retain the original
	// head stride; every newly decoded cell retains its own step/head identity.
	for i, id := range current {
		b := m.Buffers[id]
		if b == nil || b.State != "ready" {
			return nil, fmt.Errorf("KV not ready after decode")
		}
		for h := int64(0); h < modelHeads; h++ {
			for token := int64(0); token < n; token++ {
				want := Cell{uint64(i + 1), (h*c.Length + token) * modelHeadTokenBytes, true}
				if token >= c.Length {
					step := int(token-c.Length) + 1
					want = Cell{uint64(1000 + step*modelTensors + i), h * modelHeadTokenBytes, true}
				}
				if got := b.Cells[h*n+token]; got != want {
					return nil, fmt.Errorf("KV lineage mismatch tensor=%d head=%d token=%d", i, h, token)
				}
				result.KVCellsVerified++
			}
		}
	}
	if err = stage("request_cleanup_gc_and_sync", "cpu", c.CleanupNS, func() error {
		for _, id := range current {
			if err := m.Free(id); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	result.CleanupNS = c.CleanupNS
	result.ProcessNS = e.NowNS
	result.PeakUsed = m.Peak
	// The measured process keeps weights and cached pinned allocations alive.
	// FinalUsed reports these physical survivors rather than fabricating a free.
	if err = m.Check(); err != nil {
		return nil, err
	}
	result.FinalUsed = m.Used
	return result, nil
}
