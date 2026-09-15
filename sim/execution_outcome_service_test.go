package sim

import "testing"

type outcomeCostFunc func(DecisionView, DecisionPlan, ExecutionOutcome) (DecisionCostEstimate, error)

func (f outcomeCostFunc) EstimateExecutionOutcome(v DecisionView, p DecisionPlan, o ExecutionOutcome) (DecisionCostEstimate, error) {
	return f(v, p, o)
}

func TestExecutionOutcomeUsesActualGrantsAndDelaysFeedback(t *testing.T) {
	s, records := serviceFixture(t, LinearDecisionCost{FixedUS: 5, Provenance: "synthetic decision"})
	var calls int
	model := outcomeCostFunc(func(v DecisionView, p DecisionPlan, o ExecutionOutcome) (DecisionCostEstimate, error) {
		calls++
		if len(o.Grants) != 1 || o.Grants[0].Tokens != 3 || len(o.Work.Allocations) != 1 || len(v.Waiting) != 1 {
			t.Fatal("actual formation missing or late request leaked into view", o, v)
		}
		// Mutating every received slice must not alter the execution or record.
		o.Grants[0].Request = "forged"
		o.Work.Allocations[0].Request = "forged"
		v.Waiting[0].ID = "forged"
		if len(p.QueueOrder) > 0 {
			p.QueueOrder[0] = "forged"
		}
		return DecisionCostEstimate{ExtraUS: 7, Provenance: "synthetic actual execution"}, nil
	})
	if err := s.SetExecutionOutcomeCostModel(model); err != nil {
		t.Fatal(err)
	}
	a, b := serviceRequest("a", 0), serviceRequest("b", 8)
	s.InjectArrival(a)
	s.InjectArrival(b)
	for s.PeekNextEventTime() < 12 {
		s.ProcessNextEvent()
		if len(*records) != 0 || a.TTFTSet || a.ProgressIndex != 0 {
			t.Fatal("feedback or compute preceded actual host service")
		}
	}
	drainService(t, s)
	if calls != 2 || len(*records) != 2 || (*records)[0].Feedback.AtUS != 12 || (*records)[1].Feedback.AtUS != 34 {
		t.Fatal("execution feedback has wrong clock", *records)
	}
	if (*records)[0].Feedback.Grants[0].Request != "a" || (*records)[0].View.Waiting[0].ID != "a" || a.FirstTokenTime != 22 || b.FirstTokenTime != 36 {
		t.Fatal("cost input mutation affected execution", a, b, *records)
	}
	if s.decision.executionFeedback != nil || s.decision.executionPending != nil || s.KVCache.UsedBlocks() != 0 {
		t.Fatal("execution or feedback ownership leaked")
	}
}

func TestExecutionOutcomeCancellationPaysServiceAndQueuesCallbacks(t *testing.T) {
	s, records := serviceFixture(t, nil)
	var events []DecisionEvent
	if err := s.SetDecisionEventObserver(func(r DecisionEventRecord) { events = append(events, r.Event) }); err != nil {
		t.Fatal(err)
	}
	model := outcomeCostFunc(func(DecisionView, DecisionPlan, ExecutionOutcome) (DecisionCostEstimate, error) {
		return DecisionCostEstimate{ExtraUS: 12, Provenance: "synthetic cancellation boundary"}, nil
	})
	if err := s.SetExecutionOutcomeCostModel(model); err != nil {
		t.Fatal(err)
	}
	a := serviceRequest("a", 0)
	a.Deadline = 5
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateTimedOut || a.TTFTSet || s.Clock != 12 || s.KVCache.UsedBlocks() != 0 || len(*records) != 1 || (*records)[0].Feedback.AtUS != 12 {
		t.Fatal("cancelled execution computed, refunded cost or lost feedback", a, *records)
	}
	found := false
	for _, e := range events {
		if e.Kind == "request_timed_out" {
			found = true
			if e.OccurredUS != 5 || e.DeliveredUS != 12 {
				t.Fatal("policy callback ran during synchronous host service", e)
			}
		}
	}
	if !found || s.decision.executionFeedback != nil || s.decision.executionPending != nil || s.stepEvent != nil {
		t.Fatal("missing timeout event or undrained execution")
	}
}

func TestExecutionOutcomeZeroCostAndModelExclusivity(t *testing.T) {
	s, records := serviceFixture(t, nil)
	zero := outcomeCostFunc(func(DecisionView, DecisionPlan, ExecutionOutcome) (DecisionCostEstimate, error) {
		return DecisionCostEstimate{Provenance: "zero fee sensitivity"}, nil
	})
	if err := s.SetExecutionOutcomeCostModel(zero); err != nil {
		t.Fatal(err)
	}
	if err := s.SetExecutionCostModel(constantExecutionCost(1)); err == nil {
		t.Fatal("two execution models silently replaced one another")
	}
	a := serviceRequest("a", 0)
	s.InjectArrival(a)
	drainService(t, s)
	if a.FirstTokenTime != 10 || (*records)[0].Feedback.AtUS != 0 || s.decision.executionFeedback != nil {
		t.Fatal("zero outcome fee changed execution clock")
	}
}
