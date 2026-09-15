package kv

import (
	"fmt"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

func (f *PeerFabric) EnableInputPipeline(requests []kvruntime.InputRequest) error {
	return f.native.enableInputPipeline(requests)
}
func (n *NativePeerRuntime) enableInputPipeline(requests []kvruntime.InputRequest) error {
	if n == nil || !n.outputEnabled || n.inputs != nil || n.profile.InputPipeline == nil {
		return fmt.Errorf("input pipeline requires output copies and measured input profile")
	}
	p := n.profile.InputPipeline
	known := map[int64]bool{}
	for _, x := range p.Index {
		known[x.Count] = true
	}
	for count := int64(1); count <= n.hbmSlots; count++ {
		if !known[count] {
			return fmt.Errorf("unmeasured possible input index count %d", count)
		}
	}
	m := n.Runtime.Memory
	context := m.Buffers["native/model_context"]
	if context == nil || context.Bytes <= p.ReplacedGPUInputBytes {
		return fmt.Errorf("input baseline replacement exceeds context")
	}
	remaining := context.Bytes - p.ReplacedGPUInputBytes
	if err := m.Free(context.ID); err != nil {
		return err
	}
	if _, err := m.ReserveOpaque("native/model_context", "hbm_workspace", "opaque_model_context_excluding_explicit_request_inputs", remaining); err != nil {
		return err
	}
	if err := m.Allocate("native/model_context"); err != nil {
		return err
	}
	m.Capacity["host_input"] = 2 * p.HostIndexBytes
	var pages []string
	for id := int64(0); id < n.hbmSlots; id++ {
		pages = append(pages, n.hbmID(id))
	}
	inputs, err := kvruntime.NewRequestInputs(n.Runtime, *p, "hbm_workspace", "host_input", pages, requests)
	if err != nil {
		return err
	}
	n.inputs = inputs
	n.Runtime.Engine.Emit(kvruntime.Record{Name: "native_input_pipeline_enabled", Bytes: p.ReplacedGPUInputBytes, Reason: "replace opaque warm input footprint with physical request buffers, index H2D, next-token D2D and generation-readiness checks"})
	return nil
}

func (f *PeerFabric) EnableOutputCopies(policy string) error {
	return f.native.enableOutputCopies(policy)
}
func (n *NativePeerRuntime) enableOutputCopies(policy string) error {
	if n == nil || !n.pagedEnabled || n.outputEnabled {
		return fmt.Errorf("output copies require an idle paged runtime")
	}
	var storage, reserved int64
	for _, p := range n.profile.PagedForward {
		if p.Output == nil {
			return fmt.Errorf("missing output calibration for a paged shape")
		}
		storage = max(storage, p.Output.HostStorageBytes)
		reserved = max(reserved, p.Output.HostReservedBytes)
	}
	if storage <= 0 || reserved < storage {
		return fmt.Errorf("invalid scalar host storage")
	}
	e, m := n.Runtime.Engine, n.Runtime.Memory
	for _, resource := range []string{"copy_d2h", "copy_h2d"} {
		if err := e.ConfigureResource(resource, policy); err != nil {
			return err
		}
	}
	m.Capacity["host_output"] = reserved
	if err := m.RetainAllocator("host_output", reserved, true); err != nil {
		return err
	}
	if _, err := m.ReserveGranular("native/host_output", "host_output", "pinned_output_scalar", storage, 8); err != nil {
		return err
	}
	if err := m.Allocate("native/host_output"); err != nil {
		return err
	}
	n.outputEnabled = true
	e.Emit(kvruntime.Record{Name: "native_output_copy_enabled", Reason: "measured GPU-ready then native 8-byte D2H; directional copy resources; output observation precedes scatter and request completion"})
	return nil
}

// EnableTorchCopies opts into coefficients measured over live Torch-owned
// pointers. Keeping this explicit preserves prior standalone-CUDA experiments.
func (f *PeerFabric) EnableTorchCopies() error {
	return f.native.enableTorchCopies()
}
func (n *NativePeerRuntime) enableTorchCopies() error {
	if n == nil || !n.pagedEnabled || n.profile.TorchCopies == nil {
		return fmt.Errorf("live Torch copies require the paged backend and measured profile")
	}
	c := n.profile.TorchCopies
	if n.hbmSlots != c.GPUSlots || n.hostSlots != c.HostSlots {
		return fmt.Errorf("live Torch transfer arena shape mismatch")
	}
	points := map[string]kvruntime.TransferPoint{}
	for _, p := range c.Points {
		if p.Blocks == 1 {
			points[p.Direction] = p
		}
	}
	if len(points) != 2 {
		return fmt.Errorf("missing live Torch single-block direction")
	}
	n.points = points
	n.Runtime.Engine.Emit(kvruntime.Record{Name: "native_torch_copy_enabled", Reason: "56 native CUDA API copies per transaction over Torch-owned pointers; persistent stream/events; GPU 40 slots and pinned DRAM 64 slots; isolated service only, interference is not assumed free"})
	return nil
}

func (f *PeerFabric) EnablePagedForward() error {
	return f.native.enablePagedForward()
}
func (n *NativePeerRuntime) enablePagedForward() error {
	if n == nil || n.pagedEnabled || n.hbmSlots != 40 || len(n.profile.PagedForward) == 0 {
		return fmt.Errorf("paged forward requires the measured 40-slot arena and pipeline profile")
	}
	var before, reserved int64
	for _, p := range n.profile.PagedForward {
		before = max(before, p.BeforeAllocatedBytes)
		reserved = max(reserved, p.ReservedBytes)
	}
	arena := n.hbmSlots * 917504
	if before <= arena || reserved < before {
		return fmt.Errorf("invalid measured GPU allocation baseline")
	}
	m := n.Runtime.Memory
	m.Capacity["hbm_workspace"] = (40 << 30) - arena
	if err := m.RetainAllocator("hbm_workspace", reserved-arena, true); err != nil {
		return err
	}
	if _, err := m.ReserveOpaque("native/model_context", "hbm_workspace", "opaque_model_context_and_gpu_resident_inputs", before-arena); err != nil {
		return err
	}
	if err := m.Allocate("native/model_context"); err != nil {
		return err
	}
	for slot := int64(0); slot < n.hbmSlots; slot++ {
		if err := m.Invalidate(n.hbmID(slot)); err != nil {
			return err
		}
	}
	n.pagedContents = map[string][]kvruntime.Cell{}
	n.pagedEnabled = true
	n.Runtime.Engine.Emit(kvruntime.Record{Name: "native_paged_forward_enabled", Reason: "actual gather/forward/delta-scatter DAG; warm allocator reservation; transfer path remains native CUDA; concurrent slowdown uncalibrated"})
	return nil
}

func (n *NativePeerRuntime) requestOrigins(req *sim.Request, count int64) ([]uint64, error) {
	if count < 0 || count > int64(len(req.InputTokens)+len(req.OutputTokens)) {
		return nil, fmt.Errorf("request lineage exceeds supplied token reference")
	}
	var previous [32]byte
	origins := make([]uint64, count)
	for i := int64(0); i < count; i++ {
		token := sim.TokenID(0)
		if i < int64(len(req.InputTokens)) {
			token = req.InputTokens[i]
		} else {
			token = req.OutputTokens[i-int64(len(req.InputTokens))]
		}
		previous = hash.HashTokenLineage(previous, token)
		key := fmt.Sprintf("token/%x", previous)
		origin := n.origins[key]
		if origin == 0 {
			origin = uint64(len(n.origins)) + 1
			n.origins[key] = origin
			n.Runtime.Engine.Emit(kvruntime.Record{Name: "kv_token_identity", Source: req.ID, PrefixTokens: i, Origin: origin, PayloadKey: key, Reason: "causal_prefix_hash_over_supplied_input_and_output_reference"})
		}
		origins[i] = origin
	}
	return origins, nil
}

// BeginBatch implements sim.AsyncBatchExecutor. The simulator is woken only
// after scatter publication and HF workspace release actually finish.
func (s *PeerCache) AsyncBatchEnabled() bool {
	return s.fabric.phases != nil || s.fabric.native != nil && s.fabric.native.pagedEnabled
}
func (s *PeerCache) BeginBatch(now int64, work []sim.BatchWork, done func(int64)) (bool, error) {
	if s.fabric.phases != nil {
		return s.fabric.phases.begin(now, work, done)
	}
	n := s.fabric.native
	if n == nil || !n.pagedEnabled {
		return false, nil
	}
	if len(work) != 1 || n.forwardActive {
		return false, fmt.Errorf("native paged forward supports one active batch-one graph")
	}
	w := work[0]
	n.advance(now)
	ids := append([]int64(nil), s.RequestMap[w.Request.ID]...)
	hashes := make([]string, len(ids))
	for i, id := range ids {
		hashes[i] = s.Blocks[id].Hash
	}
	_, err := n.beginPagedBatch(w.Request, w.PrefixTokens, w.NewTokens, ids, hashes, func(result *kvruntime.PagedStepResult) {
		at := (result.CompleteNS + 999) / 1000
		n.completed = append(n.completed, func() {
			n.fabric.emit(PeerRecord{Time: at, Name: "native_forward_complete", Request: w.Request.ID, Value: at*1000 - result.CompleteNS, Duration: at - now, Counters: map[string]int64{"verified_cells": result.CellsVerified, "mapping_tasks": int64(result.MappingTasks), "workspace_peak_delta_bytes": result.AllocatedPeakDelta, "output_bytes": result.OutputBytes, "output_observed_ns": result.OutputObservedNS}})
			done(at)
		})
	})
	if err != nil {
		return false, err
	}
	n.arm(s.schedule)
	return true, nil
}

// beginPagedBatch submits only physical work. The caller owns policy time and
// completion observation; the method never advances or recursively pumps it.
func (n *NativePeerRuntime) beginPagedBatch(req *sim.Request, prefix, query int64, ids []int64, hashes []string, done func(*kvruntime.PagedStepResult)) (*kvruntime.PagedStepResult, error) {
	if n == nil || !n.pagedEnabled || n.forwardActive || req == nil {
		return nil, fmt.Errorf("paged worker requires an idle initialized runtime and request")
	}
	var point *kvruntime.PagedForwardPoint
	for i := range n.profile.PagedForward {
		p := &n.profile.PagedForward[i]
		if p.Prefix == prefix && p.Query == query {
			point = p
			break
		}
	}
	if point == nil {
		return nil, fmt.Errorf("unmeasured paged pipeline prefix=%d query=%d", prefix, query)
	}
	origins, err := n.requestOrigins(req, prefix+query)
	if err != nil {
		return nil, err
	}
	ids = append([]int64(nil), ids...)
	hashes = append([]string(nil), hashes...)
	var slots []string
	for _, id := range ids {
		slots = append(slots, n.hbmID(id))
	}
	var weights []float64
	for _, c := range n.profile.Compute {
		if c.Length == 256 && c.Mode == "resident" {
			index := 0
			if query == 1 {
				index = 1
			}
			weights = c.ForwardWeights[index]
			break
		}
	}
	n.forwardSerial++
	name := fmt.Sprintf("request/%s/step/%d", req.ID, n.forwardSerial)
	spec := kvruntime.PagedStepSpec{Inputs: n.inputs, Request: req.ID, ID: name, Slots: slots, Prefix: prefix, Query: query, TokenOrigins: origins, Weights: weights, Point: *point, WorkspacePool: "hbm_workspace"}
	if point.Output != nil {
		if !n.outputEnabled {
			return nil, fmt.Errorf("output-aware profile requires output_copies configuration")
		}
		spec.HostOutputBuffer = "native/host_output"
		spec.OutputCopyResource = "copy_d2h"
		spec.OutputStreamGroup = "request_stream"
		spec.OutputValue = point.Output.ReferenceToken
		spec.OutputReferenceKind = "internal_profile_reference"
		end := prefix + query
		if end >= req.InputLen() {
			index := end - req.InputLen()
			if index >= int64(len(req.OutputTokens)) {
				return nil, fmt.Errorf("output observation exceeds supplied reference")
			}
			spec.OutputValue = uint64(uint32(req.OutputTokens[index]))
			spec.OutputReferenceKind = "request_reference"
		} else if n.internalOutputReferences != nil {
			value, ok := n.internalOutputReferences[req.ID][fmt.Sprintf("%d/%d", prefix, query)]
			if !ok {
				return nil, fmt.Errorf("missing independent internal output for %s prefix=%d query=%d", req.ID, prefix, query)
			}
			spec.OutputValue = value
			spec.OutputReferenceKind = "internal_request_reference"
		}
	}
	spec.Done = func(result *kvruntime.PagedStepResult) {
		// Register complete reusable input blocks against the independent causal
		// token reference, never against a copy of the observed destination bytes.
		end := prefix + query
		for page, id := range ids {
			if int64((page+1)*16) > min(end, req.InputLen()) {
				break
			}
			if page >= len(hashes) {
				n.Runtime.Engine.Fail(fmt.Errorf("missing independent input block hash"))
				return
			}
			hash := hashes[page]
			if hash == "" {
				n.Runtime.Engine.Fail(fmt.Errorf("full input block has no hash"))
				return
			}
			expected := make([]kvruntime.Cell, 56*64)
			for tensor := 0; tensor < 56; tensor++ {
				for head := 0; head < 4; head++ {
					for token := 0; token < 16; token++ {
						expected[tensor*64+head*16+token] = kvruntime.Cell{Origin: origins[page*16+token], Offset: int64(tensor*4+head) * 256, Valid: true}
					}
				}
			}
			if old := n.pagedContents[hash]; old != nil {
				for i := range old {
					if old[i] != expected[i] {
						n.Runtime.Engine.Fail(fmt.Errorf("prefix hash aliases different causal KV"))
						return
					}
				}
			} else {
				n.pagedContents[hash] = expected
			}
			if err := n.checkPagedContents(hash, n.hbmID(id)); err != nil {
				n.Runtime.Engine.Fail(err)
				return
			}
		}
		n.forwardActive = false
		if done != nil {
			done(result)
		}
	}
	n.forwardActive = true
	spec.ServiceClasses = n.pagedServiceClasses
	spec.PhaseObserver = n.pagedPhaseObserver
	result, err := n.Runtime.PagedStep(spec)
	if err != nil {
		n.Runtime.Engine.Fail(err)
	}
	return result, err
}

func (n *NativePeerRuntime) checkPagedContents(hash, buffer string) error {
	expected := n.pagedContents[hash]
	b := n.Runtime.Memory.Buffers[buffer]
	if expected == nil || b == nil || b.State != "ready" || len(b.Cells) != len(expected) {
		return fmt.Errorf("unpublished or unknown paged payload %s", hash)
	}
	for i, c := range b.Cells {
		if c != expected[i] {
			return fmt.Errorf("paged payload mismatch hash=%s cell=%d", hash, i)
		}
	}
	return nil
}
func (f *PeerFabric) NativeAllocationSnapshot() map[string]map[string]int64 {
	if f.native == nil {
		return nil
	}
	m := f.native.Runtime.Memory
	out := map[string]map[string]int64{}
	for p, active := range m.Used {
		out[p] = map[string]int64{"active_bytes": active, "retained_bytes": m.Retained[p], "physical_bytes": m.PhysicalUsed(p), "peak_active_bytes": m.Peak[p], "peak_physical_bytes": m.PhysicalPeak[p]}
	}
	return out
}
