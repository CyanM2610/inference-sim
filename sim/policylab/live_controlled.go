package policylab

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

// policyWorkFeatures describes work performed by the existing controller. Hash
// counts are extracted from actual invocations, independently of host wall time.
// Pool entries match the entries visited by PeerFabric.Snapshot after Poll.
func policyWorkFeatures(reply *LiveReply, work kv.HashWork) kvruntime.ControlFeatures {
	f := kvruntime.ControlFeatures{"events": float64(reply.EventsProcessed), "records": float64(len(reply.Records)), "completed": float64(len(reply.Requests)),
		"hash_calls": float64(work.Calls), "hash_tokens": float64(work.Tokens), "hash_bytes": float64(work.Bytes), "hash_sha256_blocks": float64(work.SHA256Blocks), "pool_entries": 0}
	for _, pool := range reply.Pools {
		f["pool_entries"] += float64(pool["ready"] + pool["reserved"])
	}
	return f
}

// pythonPollBytes matches the native driver's json.dumps separators and field
// insertion order. In particular completed_copies is present even when empty.
func pythonPollBytes(p LivePoll) int {
	var b strings.Builder
	fmt.Fprintf(&b, "{\"now_us\": %d, \"completed_copies\": [", p.NowUS)
	for i, id := range p.Copies {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.FormatInt(id, 10))
	}
	b.WriteString("]")
	if p.Batch != 0 {
		fmt.Fprintf(&b, ", \"completed_batch\": %d", p.Batch)
	}
	if p.Token != nil {
		fmt.Fprintf(&b, ", \"observed_token\": %d", *p.Token)
	}
	b.WriteString("}\n")
	return b.Len()
}

// RunControlledLive drives the same real policy and physical executor through
// the measured driver/pipe/controller/worker protocol. Only physical completion
// tasks produce notifications; no reference command order or ACK time is used.
func RunControlledLive(c Config, p *kvruntime.Profile, control *kvruntime.ControlProfile, copyPolicy string, sink func(kvruntime.Record)) (*PhysicalLiveResult, error) {
	if control == nil {
		return nil, fmt.Errorf("control profile required")
	}
	l, x, err := newPhysicalController(c, p, copyPolicy, sink)
	if err != nil {
		return nil, err
	}
	countHashes := false
	for _, name := range []string{"hash_calls", "hash_tokens", "hash_bytes", "hash_sha256_blocks"} {
		if _, ok := control.Models["go_poll"].Coefficients[name]; ok {
			countHashes = true
		}
	}
	if countHashes {
		l.EnableHashWorkAccounting()
	}
	e := x.Runtime.Engine
	out := &PhysicalLiveResult{Scope: "Free-running real LiveController and shared physical DAG with measured control/IPC/driver/worker stage costs. Observed policy clock is captured before request packing; busy arrivals/completions await a subsequent round. Independent captured token values are causal references. Exact measured hardware/model scope; deterministic stage means do not establish tail or unseen-workload accuracy. Full paged contention and allocator internals remain incomplete."}
	operations := map[string]int{}
	var poll LivePoll
	finish := func(key string, task int64, token func()) {
		e.MustAdd(kvruntime.Task{Name: "controlled_physical_complete", Operation: key, Resource: "physical_result_observer", Parents: []int64{task}, Finish: func() error {
			out.Operations[operations[key]].EndNS = e.NowNS
			if token != nil {
				token()
			}
			return nil
		}})
	}
	driver, err := x.Runtime.ControlDriver(kvruntime.ControlDriverSpec{Profile: control, MaxRounds: 20000, Prepare: func(o kvruntime.DriverObservation) (kvruntime.ControlFeatures, error) {
		poll = LivePoll{NowUS: (o.TimeNS + 999) / 1000}
		for _, done := range o.Completed {
			index, ok := operations[done.ID]
			if !ok {
				return nil, fmt.Errorf("notification has no physical operation")
			}
			op := out.Operations[index]
			if op.EndNS == 0 || op.EndNS > o.TimeNS || op.Kind != done.Kind {
				return nil, fmt.Errorf("notification precedes physical completion")
			}
			if done.Kind == "copy" {
				poll.Copies = append(poll.Copies, op.ID)
			} else {
				poll.Batch = op.ID
				poll.Token = op.Token
			}
		}
		return kvruntime.ControlFeatures{"input_kib": float64(pythonPollBytes(poll)) / 1024}, nil
	}, Decide: func(o kvruntime.DriverObservation) (*kvruntime.DriverDecision, error) {
		reply, err := l.Poll(poll)
		if err != nil {
			return nil, err
		}
		features := policyWorkFeatures(reply, l.TakeHashWork())
		// The reply includes a diagnostic wall-time field. Predict its width
		// from the independently calibrated Poll service, never DES wall time.
		reply.ControlWallNS, err = control.Predict("go_poll", features)
		if err != nil {
			return nil, err
		}
		wire, err := json.Marshal(reply)
		if err != nil {
			return nil, err
		}
		decision := &kvruntime.ControlDecision{PolicyFeatures: features, ReplyFeatures: kvruntime.ControlFeatures{"reply_kib": float64(len(wire)+1) / 1024}}
		for _, command := range reply.Copies {
			command := command
			key := fmt.Sprintf("copy-%d", command.ID)
			decision.Commands = append(decision.Commands, kvruntime.ControlCommand{ID: key, Kind: "copy", Start: func() (int64, error) {
				result, err := x.StartTransfer(command)
				if err != nil {
					return 0, err
				}
				operations[key] = len(out.Operations)
				out.Operations = append(out.Operations, PhysicalLiveOperation{Kind: "copy", ID: command.ID, Command: command, StartNS: e.NowNS})
				finish(key, result.CompletionTask, nil)
				return result.CompletionTask, nil
			}})
		}
		if command := reply.Batch; command != nil {
			key := fmt.Sprintf("batch-%d", command.ID)
			decision.Commands = append(decision.Commands, kvruntime.ControlCommand{ID: key, Kind: "batch", Start: func() (int64, error) {
				result, err := x.StartBatch(command.ID, command.Request, command.Prefix, command.Query, command.Slots, command.Tokens)
				if err != nil {
					return 0, err
				}
				operations[key] = len(out.Operations)
				out.Operations = append(out.Operations, PhysicalLiveOperation{Kind: "batch", ID: command.ID, Command: command, StartNS: e.NowNS})
				finish(key, result.Step.CompletionTask, func() { out.Operations[operations[key]].Token = result.Token })
				return result.Step.CompletionTask, nil
			}})
		}
		decision.ReplyFeatures["commands"] = float64(len(decision.Commands))
		out.Rounds = append(out.Rounds, PhysicalLiveRound{AtNS: e.NowNS, ObservedNS: o.TimeNS, Poll: poll, Reply: reply})
		next := int64(-1)
		if reply.NextEventUS >= 0 {
			if reply.NextEventUS > math.MaxInt64/1000 {
				return nil, fmt.Errorf("policy event time overflow")
			}
			next = reply.NextEventUS * 1000
		}
		if reply.Done {
			out.Final = reply
		}
		return &kvruntime.DriverDecision{Control: decision, NextEventNS: next, Done: reply.Done}, nil
	}})
	out.Driver = driver
	if err != nil {
		return nil, err
	}
	if err = e.Run(); err != nil {
		return nil, err
	}
	if out.Final == nil || driver.CompleteNS == 0 {
		return nil, fmt.Errorf("controlled runtime did not finish")
	}
	if err = x.ValidateDrained(); err != nil {
		return nil, err
	}
	if err = l.ValidatePhysicalMemory(); err != nil {
		return nil, err
	}
	return out, nil
}
