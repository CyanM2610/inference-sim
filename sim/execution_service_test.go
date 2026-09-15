package sim

import (
	"math"
	"testing"
)

type constantExecutionCost int64

func (c constantExecutionCost) EstimateExecution(BatchExecutionWork, int64) (DecisionCostEstimate, error) {
	return DecisionCostEstimate{ExtraUS: int64(c), Provenance: "synthetic host execution"}, nil
}

func TestExecutionServiceOwnsEngineUntilComputeCanStart(t *testing.T) {
	s, records := serviceFixture(t, LinearDecisionCost{FixedUS: 5, Provenance: "synthetic decision"})
	if err := s.SetExecutionCostModel(constantExecutionCost(7)); err != nil {
		t.Fatal(err)
	}
	a, b := serviceRequest("a", 0), serviceRequest("b", 8)
	s.InjectArrival(a)
	s.InjectArrival(b)
	s.Schedule(&StepEvent{time: 9})
	for s.PeekNextEventTime() < 12 {
		s.ProcessNextEvent()
		if a.TTFTSet || a.ProgressIndex != 0 {
			t.Fatal("computation published during host service")
		}
	}
	if len(*records) != 1 || (*records)[0].ExecutionService.StartUS != 5 || (*records)[0].ExecutionService.EndUS != 12 || len((*records)[0].ExecutionWork.Allocations) != 1 {
		t.Fatal("missing actual work and execution interval", *records)
	}
	drainService(t, s)
	if a.FirstTokenTime != 22 || b.FirstTokenTime != 44-8 || len(*records) != 2 {
		t.Fatal("service overlap or late request entered formed batch", a.FirstTokenTime, b.FirstTokenTime, len(*records))
	}
	if s.stepEvent != nil || s.decision.executionPending != nil || s.KVCache.UsedBlocks() != 0 {
		t.Fatal("engine or resource lease leaked")
	}
}

func TestExecutionServiceCancellationPaysHostButCannotCompute(t *testing.T) {
	s, _ := serviceFixture(t, nil)
	if err := s.SetExecutionCostModel(constantExecutionCost(12)); err != nil {
		t.Fatal(err)
	}
	a := serviceRequest("a", 0)
	a.Deadline = 5
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateTimedOut || a.TTFTSet || s.Clock != 12 || s.KVCache.UsedBlocks() != 0 || s.decision.executionPending != nil || s.stepEvent != nil {
		t.Fatal("cancelled host work refunded, computed or leaked", a, s.Clock)
	}
}

func TestExecutionServiceLateArrivalSurvivesBlockedFrozenBatch(t *testing.T) {
	s, _ := serviceFixture(t, LinearDecisionCost{FixedUS: 5, Provenance: "synthetic"})
	s.KVCache = serviceBlockedStore{s.KVCache}
	if err := s.SetExecutionCostModel(constantExecutionCost(7)); err != nil {
		t.Fatal(err)
	}
	a, b := serviceRequest("blocked", 0), serviceRequest("short", 2)
	a.InputTokens = append(a.InputTokens, 4, 5)
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if b.State != StateCompleted || s.decision.executionPending != nil {
		t.Fatal("late arrival stranded behind execution service", b.State)
	}
}

func TestRestoreChoiceDecisionCostUsesVisibleChoices(t *testing.T) {
	c := LinearDecisionCost{FixedUS: 2, PerRestoreChoiceUS: 3, Provenance: "synthetic"}
	v := DecisionView{Estimates: &DecisionEstimates{Requests: []DecisionRequestEstimate{{Choices: make([]DecisionRestoreEstimate, 2)}, {Choices: make([]DecisionRestoreEstimate, 1)}}}}
	x, err := c.Estimate(v, DecisionPlan{})
	if err != nil || x.ExtraUS != 11 {
		t.Fatal(x, err)
	}
	v.Estimates = nil
	if _, err = c.Estimate(v, DecisionPlan{}); err == nil {
		t.Fatal("unknown estimates treated as zero work")
	}
	v.Estimates = &DecisionEstimates{Requests: []DecisionRequestEstimate{{Choices: make([]DecisionRestoreEstimate, 2)}}}
	c.PerRestoreChoiceUS = math.MaxInt64
	if _, err = c.Estimate(v, DecisionPlan{}); err == nil {
		t.Fatal("choice cost overflow accepted")
	}
}
