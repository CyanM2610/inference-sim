package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

type PhysicalLiveRound struct {
	ObservedNS int64      `json:"observed_ns,omitempty"`
	AtNS       int64      `json:"at_ns"`
	Poll       LivePoll   `json:"poll"`
	Reply      *LiveReply `json:"reply"`
}
type PhysicalLiveOperation struct {
	Kind    string       `json:"kind"`
	ID      int64        `json:"id"`
	Command any          `json:"command"`
	StartNS int64        `json:"start_ns"`
	EndNS   int64        `json:"end_ns"`
	Token   *sim.TokenID `json:"token,omitempty"`
}
type PhysicalLiveResult struct {
	Driver     *kvruntime.ControlDriverResult `json:"driver,omitempty"`
	Scope      string                         `json:"scope"`
	Final      *LiveReply                     `json:"final"`
	Rounds     []PhysicalLiveRound            `json:"rounds"`
	Operations []PhysicalLiveOperation        `json:"operations"`
}

// RunPhysicalLiveValidation checks the real controller against actual simulated
// copies and paged computation. Control/IPC service is deliberately absent in
// this integration test, so its timestamps are not a performance prediction.
// It provides no canned completion times or command replay to the controller.
func RunPhysicalLiveValidation(c Config, p *kvruntime.Profile, copyPolicy string, sink func(kvruntime.Record)) (*PhysicalLiveResult, error) {
	l, x, err := newPhysicalController(c, p, copyPolicy, sink)
	if err != nil {
		return nil, err
	}
	out := &PhysicalLiveResult{Scope: "Functional integration only: actual LiveController decisions drive shared physical DAGs and acknowledge actual completions. Control/IPC/driver service costs are omitted explicitly; do not interpret these timestamps as end-to-end performance predictions. Token values are supplied independent references, not numerical transformer inference."}
	e := x.Runtime.Engine
	var copies []int64
	var batch int64
	var token *sim.TokenID
	pendingCopies := map[int64]bool{}
	var pendingBatch, generation int64
	wakeAt := int64(-1)
	var wake func(int64)
	var poll func()
	completion := func(name string, task int64, done func()) {
		e.MustAdd(kvruntime.Task{Name: "physical_validation_completion", Operation: name, Resource: "physical_validation_inbox", Parents: []int64{task}, After: func() {
			done()
			wake((e.NowNS + 999) / 1000 * 1000)
		}})
	}
	wake = func(at int64) {
		if wakeAt >= 0 && wakeAt <= at {
			return
		}
		generation++
		version := generation
		wakeAt = at
		if err := e.At(at, func() {
			if version != generation {
				return
			}
			wakeAt = -1
			poll()
		}); err != nil {
			e.Fail(err)
		}
	}
	poll = func() {
		if len(out.Rounds) >= 20000 {
			e.Fail(fmt.Errorf("physical controller did not terminate"))
			return
		}
		observation := LivePoll{NowUS: (e.NowNS + 999) / 1000, Copies: copies, Batch: batch, Token: token}
		for _, id := range copies {
			if !pendingCopies[id] {
				e.Fail(fmt.Errorf("unknown physical copy completion"))
				return
			}
			delete(pendingCopies, id)
		}
		if batch != 0 {
			if pendingBatch != batch {
				e.Fail(fmt.Errorf("unknown physical batch completion"))
				return
			}
			pendingBatch = 0
		}
		copies, batch, token = nil, 0, nil
		reply, err := l.Poll(observation)
		if err != nil {
			e.Fail(err)
			return
		}
		reply.ControlWallNS = 0 // simulator execution wall time is never a service coefficient.
		out.Rounds = append(out.Rounds, PhysicalLiveRound{AtNS: e.NowNS, Poll: observation, Reply: reply})
		for _, command := range reply.Copies {
			command := command
			if len(pendingCopies) != 0 {
				e.Fail(fmt.Errorf("physical validation requires one copy gate"))
				return
			}
			pendingCopies[command.ID] = true
			result, err := x.StartTransfer(command)
			if err != nil {
				e.Fail(err)
				return
			}
			index := len(out.Operations)
			out.Operations = append(out.Operations, PhysicalLiveOperation{ID: command.ID, Kind: "copy", Command: command, StartNS: e.NowNS})
			completion(fmt.Sprintf("copy/%d", command.ID), result.CompletionTask, func() {
				out.Operations[index].EndNS = e.NowNS
				copies = append(copies, command.ID)
			})
		}
		if command := reply.Batch; command != nil {
			if pendingBatch != 0 {
				e.Fail(fmt.Errorf("multiple physical batches"))
				return
			}
			pendingBatch = command.ID
			result, err := x.StartBatch(command.ID, command.Request, command.Prefix, command.Query, command.Slots, command.Tokens)
			if err != nil {
				e.Fail(err)
				return
			}
			index := len(out.Operations)
			out.Operations = append(out.Operations, PhysicalLiveOperation{ID: command.ID, Kind: "batch", Command: command, StartNS: e.NowNS})
			completion(fmt.Sprintf("batch/%d", command.ID), result.Step.CompletionTask, func() {
				out.Operations[index].EndNS, out.Operations[index].Token = e.NowNS, result.Token
				batch, token = command.ID, result.Token
			})
		}
		if reply.Done {
			if pendingBatch != 0 || len(pendingCopies) != 0 {
				e.Fail(fmt.Errorf("controller finished with active physical workers"))
				return
			}
			out.Final = reply
			return
		}
		if reply.NextEventUS >= 0 {
			if reply.NextEventUS > math.MaxInt64/1000 {
				e.Fail(fmt.Errorf("policy wake timestamp overflow"))
				return
			}
			wake(max(e.NowNS, reply.NextEventUS*1000))
		} else if pendingBatch == 0 && len(pendingCopies) == 0 {
			e.Fail(fmt.Errorf("controller has no pending work or future event"))
		}
	}
	wake(0)
	if err = e.Run(); err != nil {
		return nil, err
	}
	if out.Final == nil {
		return nil, fmt.Errorf("physical controller stopped before finishing")
	}
	if err = x.ValidateDrained(); err != nil {
		return nil, err
	}
	if err = l.ValidatePhysicalMemory(); err != nil {
		return nil, err
	}
	return out, nil
}

func newPhysicalController(c Config, p *kvruntime.Profile, copyPolicy string, sink func(kvruntime.Record)) (*LiveController, *kv.PagedExecutor, error) {
	if err := c.Validate(); err != nil {
		return nil, nil, err
	}
	if len(c.Instances) != 1 || len(c.Pools) != 1 {
		return nil, nil, fmt.Errorf("physical validation requires one instance and DRAM pool")
	}
	var requests []*sim.Request
	initial := c
	initial.Requests = append([]RequestConfig(nil), c.Requests...)
	for i, r := range c.Requests {
		requests = append(requests, &sim.Request{ID: r.ID, InputTokens: r.Input, OutputTokens: r.Output})
		initial.Requests[i].Output = make([]sim.TokenID, len(r.Output))
	}
	l, err := NewLiveController(initial)
	if err != nil {
		return nil, nil, err
	}
	x, err := kv.NewPagedExecutor(kv.PagedExecutorConfig{Profile: p, HBMSlots: c.Instances[0].HBMBlocks, DRAMSlots: c.Pools[0].CapacityBlocks, CopyEnginePolicy: copyPolicy, Requests: requests}, sink)
	if err != nil {
		return nil, nil, err
	}
	if err = l.AttachPhysicalMemory(x); err != nil {
		return nil, nil, err
	}
	return l, x, nil
}
