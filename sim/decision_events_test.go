package sim

import (
	"fmt"
	"testing"
)

type eventCapPolicy struct {
	arrivals int
	events   []DecisionEvent
	order    []string
	views    []DecisionView
}

func (p *eventCapPolicy) Decide(v DecisionView) DecisionPlan {
	p.views = append(p.views, v)
	p.order = append(p.order, fmt.Sprintf("decide:%d", v.NowUS))
	cap := int64(1)
	if p.arrivals > 1 {
		cap = 4
	}
	base, _ := NewQueueDecisionPolicy("fcfs", cap)
	return base.Decide(v)
}
func (p *eventCapPolicy) Observe(f DecisionFeedback) {
	p.order = append(p.order, fmt.Sprintf("%s:%d", f.Status, f.AtUS))
}
func (p *eventCapPolicy) OnEvent(e DecisionEvent) {
	p.order = append(p.order, fmt.Sprintf("%s:%s:%d", e.Kind, e.Request, e.DeliveredUS))
	p.events = append(p.events, cloneDecisionEvent(e))
	if e.Kind == "request_arrived" {
		p.arrivals++
	}
	// Retaining/mutating a delivered value cannot corrupt the runtime or trace.
	if len(e.Batch) > 0 {
		e.Batch[0].ComputedTokens = -999
	}
	if e.Transfer != nil {
		e.Transfer.ID = -999
	}
}

func eventFixture(t *testing.T, cost int64) (*Simulator, *eventCapPolicy, *[]DecisionRecord, *[]DecisionEventRecord) {
	t.Helper()
	s, _ := serviceFixture(t, nil)
	p := &eventCapPolicy{}
	var decisions []DecisionRecord
	var events []DecisionEventRecord
	if err := s.SetDecisionPolicy(p, func(r DecisionRecord) { decisions = append(decisions, r) }); err != nil {
		t.Fatal(err)
	}
	if err := s.SetDecisionEventObserver(func(r DecisionEventRecord) { events = append(events, r) }); err != nil {
		t.Fatal(err)
	}
	if err := s.SetDecisionCostModel(LinearDecisionCost{FixedUS: cost, Provenance: "synthetic event test"}); err != nil {
		t.Fatal(err)
	}
	return s, p, &decisions, &events
}

func TestDecisionEventsBufferedArrivalChangesNextPlanOnly(t *testing.T) {
	s, p, decisions, events := eventFixture(t, 10)
	a, b := serviceRequest("a", 0), serviceRequest("b", 5)
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if a.State != StateCompleted || b.State != StateCompleted || len(*decisions) < 2 {
		t.Fatal("requests did not execute")
	}
	if (*decisions)[0].Plan.TokenCaps[0].Tokens != 1 || (*decisions)[1].Plan.TokenCaps[0].Tokens != 4 {
		t.Fatal("arrival feedback failed to change next plan")
	}
	var delayed bool
	for i, r := range *events {
		if r.Event.Sequence != uint64(i+1) || r.Event.OccurredUS > r.Event.DeliveredUS || r.CostCoverage != DecisionEventCostCoverage {
			t.Fatal("event sequence/time/cost evidence invalid")
		}
		if r.Event.Kind == "request_arrived" && r.Event.Request == "b" {
			delayed = r.Event.OccurredUS == 5 && r.Event.DeliveredUS == 10
		}
		for _, work := range r.Event.Batch {
			if work.ComputedTokens < 0 {
				t.Fatal("policy mutated retained trace")
			}
		}
	}
	if !delayed {
		t.Fatal("service callback was not deferred")
	}
	want := []string{"request_arrived:a:0", "decide:0", "applied:10", "request_arrived:b:10"}
	for i, value := range want {
		if p.order[i] != value {
			t.Fatalf("reentered frozen decision: %v", p.order)
		}
	}
	if !p.views[0].Capabilities.LifecycleEvents || p.views[0].Capabilities.TransferEvents {
		t.Fatal("wrong event capability declaration")
	}
}

func TestDecisionEventsSupersededFeedbackPrecedesBufferedTimeout(t *testing.T) {
	s, p, decisions, _ := eventFixture(t, 10)
	a, b := serviceRequest("a", 0), serviceRequest("b", 0)
	a.Deadline = 5
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if (*decisions)[0].Feedback.Status != "superseded" {
		t.Fatal("old plan not superseded")
	}
	for i, item := range p.order {
		if item == "superseded:10" {
			if p.order[i+1] != "request_timed_out:a:10" || p.order[i+2] != "decide:10" {
				t.Fatalf("wrong timeout/redecision ordering: %v", p.order)
			}
			return
		}
	}
	t.Fatal("missing superseded feedback")
}

type outputDelayModel struct {
	fixedStepModel
	delay int64
}

func (m outputDelayModel) OutputTokenProcessingTime() int64 { return m.delay }

func TestDecisionEventsFirstOutputWaitsForProcessingAndCanBeCancelled(t *testing.T) {
	for _, cancelled := range []bool{false, true} {
		s, _, _, events := eventFixture(t, 10)
		s.latencyModel = &outputDelayModel{fixedStepModel{stepTime: 10}, 7}
		a := serviceRequest("a", 0)
		a.InputTokens = a.InputTokens[:1]
		a.OutputTokens = []TokenID{8, 9}
		if cancelled {
			a.Deadline = 25
		}
		s.InjectArrival(a)
		var first int
		drainService(t, s)
		for _, row := range *events {
			if row.Event.Kind == "first_output" {
				first++
				if row.Event.OccurredUS != 27 || row.Event.DeliveredUS != 30 {
					t.Fatalf("output event leaked future processing or bypassed busy CPU: %+v", row)
				}
			}
		}
		if (!cancelled && first != 1) || (cancelled && first != 0) {
			t.Fatalf("cancelled=%v first=%d", cancelled, first)
		}
	}
}

func TestDecisionEventsFirstOutputIsUniqueAcrossRealPreemption(t *testing.T) {
	s, _, _, events := eventFixture(t, 0)
	k := &preemptionRetryStore{KVStore: s.KVCache}
	s.KVCache = k
	a := serviceRequest("a", 0)
	a.InputTokens = a.InputTokens[:1]
	a.OutputTokens = []TokenID{8, 9, 10}
	s.InjectArrival(a)
	armed := false
	for n := 0; s.HasPendingEvents(); n++ {
		if n > 1000 {
			t.Fatal("preemption failed to drain")
		}
		e := s.ProcessNextEvent()
		if _, ok := e.(*BatchCompleteEvent); ok && !armed {
			k.rejectRunningOnce = true
			armed = true
		}
	}
	first := 0
	for _, e := range *events {
		if e.Event.Kind == "first_output" {
			first++
			if e.Event.OccurredUS != 10 {
				t.Fatal("first observation changed after re-prefill")
			}
		}
	}
	if first != 1 || s.Metrics.PreemptionCount != 1 || a.State != StateCompleted {
		t.Fatal("preemption duplicated/lost first output")
	}
}

func TestDecisionEventsZeroOutputHasNoFirstOutput(t *testing.T) {
	s, _, _, events := eventFixture(t, 0)
	a := serviceRequest("a", 0)
	a.OutputTokens = nil
	s.InjectArrival(a)
	drainService(t, s)
	completed := 0
	for _, r := range *events {
		if r.Event.Kind == "first_output" {
			t.Fatal("zero-output request got phantom output")
		}
		if r.Event.Kind == "request_completed" {
			completed++
		}
	}
	if completed != 1 {
		t.Fatal("missing terminal event")
	}
}
