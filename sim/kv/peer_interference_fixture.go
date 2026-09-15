package kv

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

type PagedInterferenceCase struct {
	Prefix, Query                                    int64
	Direction, Mode                                  string
	Rounds                                           int
	ReferenceToken                                   uint64
	CopyCost                                         kvruntime.TransferCost
	LoopEnterNS, LoopExitNS, CopyGapNS, CopyOffsetNS int64
	Rates                                            kvruntime.PagedInterferenceRates
}

type PagedInterferenceResult struct {
	Prefix      int64                       `json:"prefix"`
	Query       int64                       `json:"query"`
	Direction   string                      `json:"direction"`
	Mode        string                      `json:"mode"`
	Rounds      int                         `json:"copy_rounds"`
	Request     kvruntime.InputRequest      `json:"reference_request"`
	Step        *kvruntime.PagedStepResult  `json:"step,omitempty"`
	Copies      []*kvruntime.TransferResult `json:"copies"`
	CopyStartNS int64                       `json:"copy_start_ns"`
	CopyEndNS   int64                       `json:"copy_end_ns"`
	FinalUsed   map[string]int64            `json:"final_used_bytes"`
	Scope       string                      `json:"scope"`
}

// RunPagedInterferenceFixture uses the real physical worker and its shared
// paged/input/copy DAG. The native probe prepares a prefix outside its measured
// window; this fixture explicitly initializes that reference state at time zero.
// It does not pretend that warm initialization is a simulated prefill service.
func RunPagedInterferenceFixture(p *kvruntime.Profile, c PagedInterferenceCase, sink func(kvruntime.Record)) (*PagedInterferenceResult, error) {
	if p == nil || c.Prefix < 0 || c.Query <= 0 || c.Prefix+c.Query > 624 || (c.Direction != "d2h" && c.Direction != "h2d") || c.Rounds <= 0 || c.LoopEnterNS < 0 || c.LoopExitNS < 0 || c.CopyGapNS < 0 || c.CopyOffsetNS < -1e9 || c.CopyOffsetNS > 1e9 {
		return nil, fmt.Errorf("invalid paged interference fixture")
	}
	if c.Mode != "compute_only" && c.Mode != "copy_only" && c.Mode != "overlap" {
		return nil, fmt.Errorf("unknown interference mode")
	}
	if err := c.CopyCost.Validate(); err != nil {
		return nil, err
	}
	if err := c.Rates.Validate(); err != nil {
		return nil, err
	}
	req := &sim.Request{ID: fmt.Sprintf("shape-%d-%d", c.Prefix, c.Query), InputTokens: make([]sim.TokenID, c.Prefix+c.Query), OutputTokens: []sim.TokenID{sim.TokenID(c.ReferenceToken), 0}}
	input := kvruntime.InputRequest{ID: req.ID, Output: []uint64{c.ReferenceToken, 0}}
	for i := range req.InputTokens {
		req.InputTokens[i] = sim.TokenID(100 + i)
		input.Prompt = append(input.Prompt, uint64(100+i))
	}
	var controller *kvruntime.PagedInterference
	phase := func(name string, active bool) error { return controller.Phase(name, active) }
	x, err := NewPagedExecutor(PagedExecutorConfig{Profile: p, HBMSlots: 40, DRAMSlots: 64, CopyEnginePolicy: "fifo", Requests: []*sim.Request{req},
		PhaseObserver: phase, IndexHostServiceClass: kvruntime.PhaseServiceClass("index_bind"),
		PagedServiceClasses: kvruntime.PagedServiceClasses{Forward: kvruntime.PhaseServiceClass("forward"), GatherSubmit: kvruntime.PhaseServiceClass("gather_prefix"), GatherKernel: kvruntime.PhaseServiceClass("gather_prefix")}}, sink)
	if err != nil {
		return nil, err
	}
	e, m := x.Runtime.Engine, x.Runtime.Memory
	controller, err = kvruntime.NewPagedInterference(e, c.Rates)
	if err != nil {
		return nil, err
	}
	var slots []int64
	for i := int64(0); i < (c.Prefix+c.Query+15)/16; i++ {
		slots = append(slots, i)
	}
	origins, err := x.native.requestOrigins(req, c.Prefix)
	if err != nil {
		return nil, err
	}
	e.Emit(kvruntime.Record{Name: "warm_prefix_reference_begin", Request: req.ID, PrefixTokens: c.Prefix, Reason: "native probe prepares this state outside measured service"})
	for page := int64(0); page < (c.Prefix+15)/16; page++ {
		id := x.native.hbmID(page)
		if err := m.BeginWrite(id); err != nil {
			return nil, err
		}
		for tensor := 0; tensor < 56; tensor++ {
			for head := int64(0); head < 4; head++ {
				for token := page * 16; token < min(c.Prefix, (page+1)*16); token++ {
					if err := m.ComputeRange(id, int64(tensor)*16384+head*4096+(token%16)*256, 256, origins[token], int64(tensor*4+int(head))*256); err != nil {
						return nil, err
					}
				}
			}
		}
		if err := m.Publish(id); err != nil {
			return nil, err
		}
		if (page+1)*16 <= c.Prefix {
			x.native.pagedContents[x.hashes[req.ID][page]] = append([]kvruntime.Cell(nil), m.Buffers[id].Cells...)
		}
	}
	source, destination := "hbm/slot/39", "dram/slot/63"
	if c.Direction == "h2d" {
		source, destination = destination, source
	}
	// Both arenas were initialized by the executor. Reuse their storage through
	// the ordinary invalidation transition before seeding diagnostic contents.
	for _, id := range []string{source, destination} {
		if err := m.Invalidate(id); err != nil {
			return nil, err
		}
	}
	if err := m.Fill(source, 3); err != nil {
		return nil, err
	}
	if err := m.Fill(destination, 0); err != nil {
		return nil, err
	}
	result := &PagedInterferenceResult{Prefix: c.Prefix, Query: c.Query, Direction: c.Direction, Mode: c.Mode, Rounds: c.Rounds, Request: input,
		Scope: "Full shared physical paged worker with explicit warm reference initialization outside the measured window. Direction-shared effective phase/host rates, physical DMA and actual dependency waits. Prior memory envelopes retained; this is not a new absolute memory-peak or serving-controller validation."}
	var nextCopy func(int)
	loopTask := func(name string, ns int64, after func()) {
		e.MustAdd(kvruntime.Task{Name: name, Operation: "background", Resource: "copy_thread", ServiceClass: kvruntime.CopyHostClass, DurationNS: ns, After: after})
	}
	nextCopy = func(index int) {
		if index == c.Rounds {
			loopTask("copy_loop_exit", c.LoopExitNS, func() { result.CopyEndNS = e.NowNS })
			return
		}
		spans := make([]kvruntime.Span, 56)
		for i := range spans {
			spans[i] = kvruntime.Span{SourceOffset: int64(i) * 16384, DestinationOffset: int64(i) * 16384, Bytes: 16384}
		}
		tx, err := x.Runtime.Transfer(kvruntime.TransferSpec{ID: fmt.Sprintf("background/%d", index), Source: source, Destination: destination, CPU: "copy_thread", DMA: "copy_" + c.Direction, StreamGroup: "kv_stream", Spans: spans, Cost: c.CopyCost, HostServiceClass: kvruntime.CopyHostClass, Activity: controller.Copy, Done: func(tx *kvruntime.TransferResult) {
			if tx.Bytes != 917504 {
				e.Fail(fmt.Errorf("wrong copy byte count"))
				return
			}
			if index+1 == c.Rounds {
				nextCopy(index + 1)
			} else {
				loopTask("copy_between_calls", c.CopyGapNS, func() { nextCopy(index + 1) })
			}
		}})
		if err != nil {
			e.Fail(err)
			return
		}
		result.Copies = append(result.Copies, tx)
	}
	batchAt, copyAt := int64(0), int64(0)
	if c.Mode == "overlap" {
		batchAt = max(0, -c.CopyOffsetNS)
		copyAt = max(0, c.CopyOffsetNS)
	}
	if c.Mode != "copy_only" {
		if err := e.At(batchAt, func() {
			b, err := x.StartBatch(1, req.ID, c.Prefix, c.Query, slots, req.InputTokens[c.Prefix:])
			if err != nil {
				e.Fail(err)
				return
			}
			result.Step = b.Step
		}); err != nil {
			return nil, err
		}
	}
	if c.Mode != "compute_only" {
		if err := e.At(copyAt, func() {
			result.CopyStartNS = e.NowNS
			loopTask("copy_loop_enter", c.LoopEnterNS, func() { nextCopy(0) })
		}); err != nil {
			return nil, err
		}
	}
	if err := e.Run(); err != nil {
		return nil, err
	}
	if err := x.ValidateDrained(); err != nil {
		return nil, err
	}
	if err := controller.CheckIdle(); err != nil {
		return nil, err
	}
	if c.Mode != "compute_only" {
		for _, cell := range m.Buffers[destination].Cells {
			if !cell.Valid || cell.Origin != 3 {
				return nil, fmt.Errorf("background copy contents changed")
			}
		}
	}
	result.FinalUsed = m.Used
	return result, nil
}
