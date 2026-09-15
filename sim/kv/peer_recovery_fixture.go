package kv

import (
	"fmt"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

type RecoveryCase struct {
	Length                   int64                        `json:"length"`
	Mode                     string                       `json:"mode"`
	Requests                 []kvruntime.InputRequest     `json:"requests"`
	InternalOutputReferences map[string]map[string]uint64 `json:"internal_output_references"`
}
type RecoveryResult struct {
	Case                RecoveryCase     `json:"case"`
	Actions             []map[string]any `json:"actions"`
	ObservedOutputs     map[string]int64 `json:"observed_outputs"`
	RejectedEarlyAccess int              `json:"rejected_early_access"`
	FinalUsed           map[string]int64 `json:"final_used_bytes"`
	CompleteNS          int64            `json:"complete_ns"`
	Scope               string           `json:"scope"`
}

// RunRecoveryFixture generates its own physical sequence from semantic case
// parameters. Native timestamps/ACKs are not inputs. The pause is scripted;
// automatic scheduler/controller behavior is outside this component fixture.
func RunRecoveryFixture(p *kvruntime.Profile, c RecoveryCase, sink func(kvruntime.Record)) (*RecoveryResult, error) {
	validMode := map[string]bool{"resident": true, "restore_prefill": true, "drop_prefill": true, "restore_decode": true, "drop_decode": true}
	if p == nil || !validMode[c.Mode] || (c.Length != 129 && c.Length != 257 && c.Length != 385) || len(c.Requests) != 2 || c.InternalOutputReferences == nil {
		return nil, fmt.Errorf("invalid recovery case or missing independent references")
	}
	requests := map[string]*sim.Request{}
	var list []*sim.Request
	for _, r := range c.Requests {
		if requests[r.ID] != nil || (r.ID != "primary" && r.ID != "replacement") || int64(len(r.Prompt)) != c.Length || len(r.Output) != map[string]int{"primary": 8, "replacement": 1}[r.ID] {
			return nil, fmt.Errorf("invalid recovery request")
		}
		req := &sim.Request{ID: r.ID}
		for _, v := range r.Prompt {
			req.InputTokens = append(req.InputTokens, sim.TokenID(v))
		}
		for _, v := range r.Output {
			req.OutputTokens = append(req.OutputTokens, sim.TokenID(v))
		}
		requests[r.ID] = req
		list = append(list, req)
	}
	x, err := NewPagedExecutor(PagedExecutorConfig{Profile: p, HBMSlots: 40, DRAMSlots: 64, CopyEnginePolicy: "fifo", Requests: list, InternalOutputReferences: c.InternalOutputReferences}, sink)
	if err != nil {
		return nil, err
	}
	e := x.Runtime.Engine
	result := &RecoveryResult{Case: c, ObservedOutputs: map[string]int64{"primary": 0, "replacement": 0}, Scope: "Scripted physical checkpoint/loss/overwrite/remap/recompute with real shared worker and supplied scalar/token references. No native timestamps used for scheduling; control-fixture administration is untimed. Prior service/memory profiles retained, no automatic scheduler or new allocator validation."}
	var slots, remapped []int64
	for i := int64(0); i < (c.Length+7+15)/16; i++ {
		slots = append(slots, i)
		remapped = append(remapped, (7*i+17)%40)
	}
	var serial int64
	batch := func(request string, prefix, query int64, mapping []int64) (int64, error) {
		serial++
		req := requests[request]
		tokens := append(append([]sim.TokenID(nil), req.InputTokens...), req.OutputTokens...)
		bound := append([]int64(nil), mapping[:(prefix+query+15)/16]...)
		body, err := x.StartBatch(serial, request, prefix, query, bound, tokens[prefix:prefix+query])
		if err != nil {
			return 0, err
		}
		if err = e.Run(); err != nil {
			return 0, err
		}
		if body.Step.OutputIndex != nil {
			index := *body.Step.OutputIndex
			if body.Step.RecomputedOutput {
				if index >= result.ObservedOutputs[request] {
					return 0, fmt.Errorf("recompute advanced observation frontier")
				}
			} else {
				if index != result.ObservedOutputs[request] {
					return 0, fmt.Errorf("new output skipped observation frontier")
				}
				result.ObservedOutputs[request]++
			}
		}
		result.Actions = append(result.Actions, map[string]any{"kind": "batch", "command": map[string]any{"id": serial, "request": request, "prefix": prefix, "query": query, "slots": bound, "tokens": tokens[prefix : prefix+query]}, "step": body.Step, "token": x.Runtime.Memory.Buffers["native/host_output"].Cells[0].Origin})
		return prefix + query, nil
	}
	transfer := func(direction string, page int64, mapping []int64) error {
		serial++
		hash := x.hashes["primary"][page]
		source, destination, reason := x.native.hbmID(page), fmt.Sprintf("dram/slot/%d", page), "recovery_store"
		if direction == "h2d" {
			source, destination, reason = destination, x.native.hbmID(mapping[page]), "recovery_restore"
		}
		tx, err := x.StartTransfer(ExternalTransfer{ID: serial, Hash: hash, Source: source, Destination: destination, Reason: reason, Bytes: 917504})
		if err != nil {
			return err
		}
		if page == 0 {
			var rejected error
			if direction == "d2h" {
				rejected = x.InvalidateHBM(page)
			} else {
				rejected = x.CheckContents(hash, destination)
			}
			if rejected == nil {
				return fmt.Errorf("in-flight recovery storage became accessible")
			}
			result.RejectedEarlyAccess++
			e.Emit(kvruntime.Record{Name: "recovery_guard_rejected", Operation: tx.ID, Source: source, Destination: destination, Reason: rejected.Error()})
		}
		if err = e.Run(); err != nil {
			return err
		}
		if err = x.CheckTransfer(serial, hash, destination); err != nil {
			return err
		}
		result.Actions = append(result.Actions, map[string]any{"kind": "copy", "direction": direction, "command": map[string]any{"id": serial, "source": source, "destination": destination, "bytes": 917504}, "transfer": tx})
		return nil
	}
	invalidate := func(mapping []int64, reason string) error {
		for _, slot := range mapping {
			if err := x.InvalidateHBM(slot); err != nil {
				return err
			}
		}
		result.Actions = append(result.Actions, map[string]any{"kind": "invalidate", "slots": append([]int64(nil), mapping...), "reason": reason, "time_ns": e.NowNS})
		return nil
	}
	finish := func(start int64, mapping []int64) (int64, error) {
		for start < c.Length {
			var err error
			start, err = batch("primary", start, min(int64(128), c.Length-start), mapping)
			if err != nil {
				return 0, err
			}
		}
		for start < c.Length+7 {
			var err error
			start, err = batch("primary", start, 1, mapping)
			if err != nil {
				return 0, err
			}
		}
		return start, nil
	}
	var computed int64
	if c.Mode == "resident" {
		computed, err = finish(0, slots)
	} else {
		computed, err = batch("primary", 0, 128, slots)
		if err != nil {
			return nil, err
		}
		if strings.HasSuffix(c.Mode, "decode") {
			for computed < c.Length {
				computed, err = batch("primary", computed, min(int64(128), c.Length-computed), slots)
				if err != nil {
					return nil, err
				}
			}
			for result.ObservedOutputs["primary"] < 3 {
				computed, err = batch("primary", computed, 1, slots)
				if err != nil {
					return nil, err
				}
			}
		}
		checkpoint := min(computed, c.Length) / 16
		result.Actions = append(result.Actions, map[string]any{"kind": "preempt", "computed_tokens": computed, "observed_outputs": result.ObservedOutputs["primary"], "checkpoint_pages": checkpoint, "time_ns": e.NowNS})
		if strings.HasPrefix(c.Mode, "restore") {
			for page := int64(0); page < checkpoint; page++ {
				if err = transfer("d2h", page, slots); err != nil {
					return nil, err
				}
			}
		}
		if err = invalidate(slots, "release_primary_and_reuse"); err != nil {
			return nil, err
		}
		for other := int64(0); other < c.Length; {
			other, err = batch("replacement", other, min(int64(128), c.Length-other), slots)
			if err != nil {
				return nil, err
			}
		}
		for page := int64(0); page < checkpoint; page++ {
			if err = x.CheckContents(x.hashes["replacement"][page], x.native.hbmID(page)); err != nil {
				return nil, err
			}
			if x.CheckContents(x.hashes["primary"][page], x.native.hbmID(page)) == nil {
				return nil, fmt.Errorf("old primary prefix survived replacement")
			}
		}
		result.Actions = append(result.Actions, map[string]any{"kind": "overwrite_verified", "logical_tokens": checkpoint * 16, "time_ns": e.NowNS})
		if err = invalidate(remapped, "new_primary_mapping"); err != nil {
			return nil, err
		}
		start := int64(0)
		if strings.HasPrefix(c.Mode, "restore") {
			for page := int64(0); page < checkpoint; page++ {
				if err = transfer("h2d", page, remapped); err != nil {
					return nil, err
				}
			}
			start = checkpoint * 16
		}
		computed, err = finish(start, remapped)
	}
	if err != nil {
		return nil, err
	}
	if computed != c.Length+7 || result.ObservedOutputs["primary"] != 8 {
		return nil, fmt.Errorf("recovery did not reach final output")
	}
	if err = x.ValidateDrained(); err != nil {
		return nil, err
	}
	result.CompleteNS = e.NowNS
	result.FinalUsed = x.Runtime.Memory.Used
	return result, nil
}
