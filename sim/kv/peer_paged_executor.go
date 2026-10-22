package kv

import (
	"fmt"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

// PagedExecutor is the physical worker behind externally acknowledged commands.
// It shares the legacy native adapter's arena setup and paged implementation,
// but owns no scheduler, policy clock, or synthetic completion acknowledgement.
type PagedExecutor struct {
	copyHostClass                               string
	copyActivity                                func(bool) error
	copySubmitClass, copyD2HClass, copyH2DClass string
	Runtime                                     *kvruntime.Runtime
	native                                      *NativePeerRuntime
	requests                                    map[string]*sim.Request
	hashes                                      map[string][]string
	batches                                     map[int64]bool
	transfers                                   map[int64]*physicalTransferState
}

type physicalTransferState struct {
	hash, destination string
	complete          bool
}

type PagedExecutorConfig struct {
	// Optional independent scalar references for internal prefill, keyed by
	// request and "prefix/query". When supplied, every executed internal shape
	// must have a reference; a shape-profile token is not an input-dependent one.
	InternalOutputReferences                    map[string]map[string]uint64
	PhaseObserver                               func(string, bool) error
	CopyActivity                                func(bool) error
	IndexHostServiceClass, CopyHostServiceClass string
	// Optional bindings for an external, calibrated rate model. Empty preserves
	// the solo-cost path; setting names alone does not imply measured slowdown.
	PagedServiceClasses                                              kvruntime.PagedServiceClasses
	CopySubmitServiceClass, CopyD2HServiceClass, CopyH2DServiceClass string
	Profile                                                          *kvruntime.Profile
	NUMA                                                             int
	HBMSlots, DRAMSlots                                              int64
	CopyEnginePolicy                                                 string
	Requests                                                         []*sim.Request // independent token references, never policy output budgets
}

type PagedBatchResult struct {
	Step  *kvruntime.PagedStepResult
	Token *sim.TokenID // after physical completion; Step.RecomputedOutput distinguishes old observations
}

func NewPagedExecutor(c PagedExecutorConfig, sink func(kvruntime.Record)) (*PagedExecutor, error) {
	if c.NUMA != 0 || c.HBMSlots != 40 || c.DRAMSlots != 64 || len(c.Requests) == 0 || (c.CopyEnginePolicy != "fifo" && c.CopyEnginePolicy != "stream_drain") {
		return nil, fmt.Errorf("paged executor requires measured NUMA0, 40/64 arenas, references and copy policy")
	}
	n, err := newNativePeerRuntime(c.Profile, c.NUMA, c.HBMSlots, c.DRAMSlots, sink)
	if err != nil {
		return nil, err
	}
	x := &PagedExecutor{Runtime: n.Runtime, native: n, requests: map[string]*sim.Request{}, hashes: map[string][]string{}, batches: map[int64]bool{}, transfers: map[int64]*physicalTransferState{}}
	n.pagedServiceClasses = c.PagedServiceClasses
	n.pagedPhaseObserver = c.PhaseObserver
	x.copyHostClass, x.copyActivity = c.CopyHostServiceClass, c.CopyActivity
	x.copySubmitClass, x.copyD2HClass, x.copyH2DClass = c.CopySubmitServiceClass, c.CopyD2HServiceClass, c.CopyH2DServiceClass
	var inputs []kvruntime.InputRequest
	for _, req := range c.Requests {
		if req == nil || req.ID == "" || x.requests[req.ID] != nil || len(req.InputTokens) == 0 || len(req.OutputTokens) == 0 {
			return nil, fmt.Errorf("invalid/duplicate physical request reference")
		}
		r := &sim.Request{ID: req.ID, InputTokens: append([]sim.TokenID(nil), req.InputTokens...), OutputTokens: append([]sim.TokenID(nil), req.OutputTokens...)}
		x.requests[r.ID] = r
		x.hashes[r.ID] = hash.ComputeBlockHashes(16, r.InputTokens)
		input := kvruntime.InputRequest{ID: r.ID}
		for _, token := range r.InputTokens {
			if token < 0 {
				return nil, fmt.Errorf("negative input token")
			}
			input.Prompt = append(input.Prompt, uint64(token))
		}
		for _, token := range r.OutputTokens {
			if token < 0 {
				return nil, fmt.Errorf("negative output token")
			}
			input.Output = append(input.Output, uint64(token))
		}
		inputs = append(inputs, input)
	}
	if c.InternalOutputReferences != nil {
		n.internalOutputReferences = map[string]map[string]uint64{}
		for request, refs := range c.InternalOutputReferences {
			req := x.requests[request]
			if req == nil {
				return nil, fmt.Errorf("unknown internal-output reference request %s", request)
			}
			n.internalOutputReferences[request] = map[string]uint64{}
			for shape, token := range refs {
				found := false
				for _, point := range c.Profile.PagedForward {
					if shape == fmt.Sprintf("%d/%d", point.Prefix, point.Query) && point.Prefix+point.Query < req.InputLen() {
						found = true
						break
					}
				}
				if !found {
					return nil, fmt.Errorf("invalid internal-output reference shape %s", shape)
				}
				n.internalOutputReferences[request][shape] = token
			}
		}
	}
	if err = n.enablePagedForward(); err != nil {
		return nil, err
	}
	if err = n.enableTorchCopies(); err != nil {
		return nil, err
	}
	if err = n.enableOutputCopies(c.CopyEnginePolicy); err != nil {
		return nil, err
	}
	if err = n.enableInputPipeline(inputs); err != nil {
		return nil, err
	}
	n.inputs.IndexHostServiceClass = c.IndexHostServiceClass
	n.Runtime.Engine.Emit(kvruntime.Record{Name: "external_paged_executor_enabled", Reason: "shared physical primitives; policy acknowledgements remain caller-owned; supplied tokens are causal references, not numerical model predictions"})
	return x, nil
}

func (x *PagedExecutor) StartBatch(id int64, request string, prefix, query int64, slots []int64, tokens []sim.TokenID) (*PagedBatchResult, error) {
	req := x.requests[request]
	if id <= 0 || x.batches[id] || req == nil || prefix < 0 || query <= 0 || query != int64(len(tokens)) || prefix > int64(len(req.InputTokens)+len(req.OutputTokens))-query {
		return nil, fmt.Errorf("invalid/duplicate physical batch command")
	}
	for i, token := range tokens {
		at := prefix + int64(i)
		var want sim.TokenID
		if at < req.InputLen() {
			want = req.InputTokens[at]
		} else {
			want = req.OutputTokens[at-req.InputLen()]
		}
		if token != want {
			return nil, fmt.Errorf("command consumes wrong input token at %d", at)
		}
	}
	out := &PagedBatchResult{}
	step, err := x.native.beginPagedBatch(req, prefix, query, slots, x.hashes[request], func(result *kvruntime.PagedStepResult) {
		if index := prefix + query - req.InputLen(); index >= 0 {
			// PagedStep has already checked the physical scalar and next-input
			// copies. The reference is exposed only at its actual completion.
			token := req.OutputTokens[index]
			out.Token = &token
		}
	})
	if err != nil {
		return nil, err
	}
	x.batches[id] = true
	out.Step = step
	return out, nil
}

func (x *PagedExecutor) StartTransfer(c ExternalTransfer) (*kvruntime.TransferResult, error) {
	if c.ID <= 0 || x.transfers[c.ID] != nil || c.Bytes != 917504 || c.Hash == "" {
		return nil, fmt.Errorf("invalid physical transfer command")
	}
	direction := ""
	if strings.HasPrefix(c.Source, "hbm/slot/") && strings.HasPrefix(c.Destination, "dram/slot/") {
		direction = "d2h"
	}
	if strings.HasPrefix(c.Source, "dram/slot/") && strings.HasPrefix(c.Destination, "hbm/slot/") {
		direction = "h2d"
	}
	if direction == "" {
		return nil, fmt.Errorf("unmeasured transfer direction")
	}
	if err := x.CheckContents(c.Hash, c.Source); err != nil {
		return nil, err
	}
	spans := make([]kvruntime.Span, 56)
	for i := range spans {
		spans[i] = kvruntime.Span{SourceOffset: int64(i) * 16384, DestinationOffset: int64(i) * 16384, Bytes: 16384}
	}
	state := &physicalTransferState{hash: c.Hash, destination: c.Destination}
	dmaClass := x.copyD2HClass
	if direction == "h2d" {
		dmaClass = x.copyH2DClass
	}
	result, err := x.Runtime.Transfer(kvruntime.TransferSpec{ID: fmt.Sprintf("peer/%d/%s", c.ID, c.Reason), Source: c.Source, Destination: c.Destination, CPU: "copy_thread", DMA: "copy_" + direction, StreamGroup: "kv_stream", Spans: spans, Cost: x.native.points[direction].Cost, SubmitServiceClass: x.copySubmitClass, DMAServiceClass: dmaClass, HostServiceClass: x.copyHostClass, Activity: x.copyActivity, Done: func(*kvruntime.TransferResult) {
		if err := x.CheckContents(c.Hash, c.Destination); err != nil {
			x.Runtime.Engine.Fail(err)
			return
		}
		state.complete = true
	}})
	if err == nil {
		x.transfers[c.ID] = state
	}
	return result, err
}

func (x *PagedExecutor) InvalidateHBM(slot int64) error {
	if slot < 0 || slot >= x.native.hbmSlots {
		return fmt.Errorf("unknown physical HBM slot")
	}
	return x.Runtime.Memory.Invalidate(x.native.hbmID(slot))
}
func (x *PagedExecutor) CheckContents(hash, buffer string) error {
	return x.native.checkPagedContents(hash, buffer)
}
func (x *PagedExecutor) CheckTransfer(id int64, hash, buffer string) error {
	state := x.transfers[id]
	if state == nil || !state.complete || state.hash != hash || state.destination != buffer {
		return fmt.Errorf("physical transfer %d has not completed for this destination/payload", id)
	}
	return x.CheckContents(hash, buffer)
}
func (x *PagedExecutor) CheckIdle(buffer string) error {
	b := x.Runtime.Memory.Buffers[buffer]
	if b == nil || b.Pins != 0 || b.Writers != 0 || b.Holds != 0 {
		return fmt.Errorf("physical buffer %s remains active", buffer)
	}
	return nil
}
func (x *PagedExecutor) ValidateDrained() error {
	if x.native.forwardActive || x.Runtime.Engine.Pending() != 0 {
		return fmt.Errorf("physical pipeline is not drained")
	}
	if err := x.Runtime.Memory.Check(); err != nil {
		return err
	}
	for id := range x.Runtime.Memory.Buffers {
		if err := x.CheckIdle(id); err != nil {
			return err
		}
	}
	return x.Runtime.Engine.Error()
}
