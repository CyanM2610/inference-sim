package sim

import (
	"fmt"
	"time"
)

const DecisionEventCostCoverage = "host_measured_callback_sim_time_uncalibrated"

// DecisionProgress reports completed work, never predicted output lengths.
type DecisionProgress struct {
	Request        string `json:"request"`
	ComputedTokens int64  `json:"computed_tokens"`
	GrantedTokens  int64  `json:"granted_tokens"`
}

// DecisionEvent is a detached fact. Sequence orders occurrence within an
// instance; DeliveredUS may be later when CPU decision service is busy.
// request_completed means scheduler completion/resource release, not client
// response delivery. batch_returned is an engine boundary, not a GPU timestamp.
type DecisionEvent struct {
	Sequence    uint64             `json:"sequence"`
	Kind        string             `json:"kind"`
	OccurredUS  int64              `json:"occurred_us"`
	DeliveredUS int64              `json:"delivered_us"`
	Request     string             `json:"request,omitempty"`
	Batch       []DecisionProgress `json:"batch,omitempty"`
	Transfer    *DecisionTransfer  `json:"transfer,omitempty"`
	Reason      string             `json:"reason,omitempty"`
}

type DecisionEventRecord struct {
	Instance       string        `json:"instance,omitempty"`
	Event          DecisionEvent `json:"event"`
	CallbackWallNS int64         `json:"callback_wall_ns"`
	CostCoverage   string        `json:"cost_coverage"`
}

// DecisionEventListener is optional; policies without it retain Decide/Observe.
type DecisionEventListener interface{ OnEvent(DecisionEvent) }

// DecisionEventSource is implemented by trusted backends. They publish only
// adopted facts, after their completion callback has updated resource state.
type DecisionEventSource interface{ BindDecisionEvents(func(DecisionEvent)) }

type decisionEvents struct {
	listener    DecisionEventListener
	observe     func(DecisionEventRecord)
	queue       []DecisionEvent
	sequence    uint64
	delivering  bool
	firstOutput map[string]bool
	cancelled   map[string]bool
}

func (s *Simulator) enableDecisionEvents() error {
	if s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents {
		return fmt.Errorf("decision events require a policy and batch completion events before execution")
	}
	if s.decision.events == nil {
		s.decision.events = &decisionEvents{firstOutput: map[string]bool{}, cancelled: map[string]bool{}}
		if source, ok := s.KVCache.(DecisionEventSource); ok {
			source.BindDecisionEvents(s.notifyDecisionEvent)
		}
	}
	return nil
}

// SetDecisionEventObserver enables trace-only observation even without a policy
// listener. It does not charge a modeled callback cost or change batch policy.
func (s *Simulator) SetDecisionEventObserver(observe func(DecisionEventRecord)) error {
	if err := s.enableDecisionEvents(); err != nil {
		return err
	}
	s.decision.events.observe = observe
	return nil
}

func cloneDecisionEvent(e DecisionEvent) DecisionEvent {
	e.Batch = append([]DecisionProgress(nil), e.Batch...)
	if e.Transfer != nil {
		transfer := *e.Transfer
		e.Transfer = &transfer
	}
	return e
}

func (s *Simulator) notifyDecisionEvent(e DecisionEvent) {
	if e.Kind == "request_completed" || e.Kind == "request_timed_out" {
		s.noteDecisionControlCompletion(e.OccurredUS)
	}
	if s.decision == nil || s.decision.events == nil {
		return
	}
	d := s.decision.events
	if e.Kind == "request_timed_out" {
		d.cancelled[e.Request] = true
	}
	if e.Kind == "first_output" && d.cancelled[e.Request] {
		return
	}
	d.sequence++
	e.Sequence = d.sequence
	e.DeliveredUS = 0
	d.queue = append(d.queue, cloneDecisionEvent(e))
	if s.decision.pending == nil && !s.decision.applying {
		s.flushDecisionEvents(e.OccurredUS)
	}
}

func (s *Simulator) flushDecisionEvents(now int64) {
	if s.decision == nil || s.decision.events == nil {
		return
	}
	if s.decision.executionOutcomeCost != nil && s.decision.executionPending != nil {
		return // the native controller cannot run callbacks during this CPU service
	}
	d := s.decision.events
	if d.delivering {
		return
	}
	d.delivering = true
	defer func() { d.delivering = false }()
	for len(d.queue) > 0 {
		e := d.queue[0]
		d.queue = d.queue[1:]
		if e.OccurredUS > now {
			panic("future policy event delivery")
		}
		e.DeliveredUS = now
		wall := int64(0)
		if d.listener != nil {
			start := time.Now()
			d.listener.OnEvent(cloneDecisionEvent(e))
			wall = time.Since(start).Nanoseconds()
		}
		if d.observe != nil {
			d.observe(DecisionEventRecord{Event: e, CallbackWallNS: wall, CostCoverage: DecisionEventCostCoverage})
		}
	}
}

func (s *Simulator) decisionBatchReturned(at int64) {
	if s.decision == nil || s.decision.events == nil {
		return
	}
	e := DecisionEvent{Kind: "batch_returned", OccurredUS: at}
	var first []DecisionEvent
	for _, r := range s.RunningBatch.Requests {
		if r.NumNewTokens <= 0 {
			continue
		}
		e.Batch = append(e.Batch, DecisionProgress{Request: r.ID, ComputedTokens: r.ProgressIndex, GrantedTokens: int64(r.NumNewTokens)})
		// TTFTSet may reset on re-prefill; the subscription remembers the first
		// actual output independently, just like policylab's first observation.
		if r.TTFTSet && len(r.OutputTokens) > 0 && !s.decision.events.firstOutput[r.ID] {
			s.decision.events.firstOutput[r.ID] = true
			first = append(first, DecisionEvent{Kind: "first_output", Request: r.ID, OccurredUS: r.ArrivalTime + r.FirstTokenTime})
		}
	}
	s.notifyDecisionEvent(e)
	for _, e := range first {
		if e.OccurredUS > at {
			s.Schedule(&decisionNotificationEvent{event: e})
		} else {
			e.OccurredUS = at
			s.notifyDecisionEvent(e)
		}
	}
}

// Only future output delivery uses an extra event. Normal zero-overhead
// notifications are emitted inline after the existing state transition.
type decisionNotificationEvent struct{ event DecisionEvent }

func (e *decisionNotificationEvent) Timestamp() int64     { return e.event.OccurredUS }
func (e *decisionNotificationEvent) Priority() int        { return PriorityAdapterLoad }
func (e *decisionNotificationEvent) Execute(s *Simulator) { s.notifyDecisionEvent(e.event) }
