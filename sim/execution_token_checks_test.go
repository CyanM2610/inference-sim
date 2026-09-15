package sim

import "testing"

func TestTokenChecksKeepHeldResidentsAndExcludeUnvisitedBudgetTail(t *testing.T) {
	s, _, records := residentWaitFixture(t, 0, pauseLong)
	if err := s.EnableExecutionWorkObservation(); err != nil {
		t.Fatal(err)
	}
	s.InjectArrival(longWaitRequest())
	drainService(t, s)
	if len(*records) != 4 {
		t.Fatal("unexpected wait execution")
	}
	for i, r := range *records {
		if r.ExecutionWork == nil || !r.ExecutionWork.TokenChecksKnown || len(r.ExecutionWork.TokenChecks) != 1 {
			t.Fatal("missing actual token check", i, r.ExecutionWork)
		}
		check := r.ExecutionWork.TokenChecks[0]
		if check.Request != "long" || check.AllocationStart != 0 || check.Held != (i == 1) {
			t.Fatal("incorrect wait/check identity", check)
		}
		if i == 1 && (check.Tokens != 0 || len(r.ExecutionWork.Allocations) != 0) {
			t.Fatal("held check allocated or granted tokens", r.ExecutionWork)
		}
	}
	other, _ := serviceFixture(t, nil)
	a, b := longWaitRequest(), longWaitRequest()
	a.ID, b.ID = "first", "unvisited"
	a.State, b.State = StateRunning, StateRunning
	formation := &VLLMBatchFormation{}
	result := formation.FormBatch(BatchContext{RunningBatch: &Batch{Requests: []*Request{a, b}},
		WaitQ: &WaitQueue{}, KVCache: other.KVCache, MaxNumBatchedTokens: 2, MaxNumSeqs: 2,
		PrefillTokenThreshold: 2, ComputedTokens: map[string]int64{}, ObserveExecutionWork: true})
	if len(result.ExecutionWork.TokenChecks) != 1 || result.ExecutionWork.TokenChecks[0].Request != "first" || b.NumNewTokens != 0 {
		t.Fatal("budget-exhausted tail counted as a check", result.ExecutionWork)
	}
}

type changingTokenCheckCost struct{}

func (changingTokenCheckCost) EstimateExecution(w BatchExecutionWork, _ int64) (DecisionCostEstimate, error) {
	if len(w.TokenChecks) > 0 {
		w.TokenChecks[0].Request = "changed"
		w.TokenChecks[0].Tokens = 999
	}
	return DecisionCostEstimate{Provenance: "test detached work"}, nil
}

func TestExecutionPriceCannotRewriteObservedTokenChecks(t *testing.T) {
	s, records := serviceFixture(t, nil)
	if err := s.SetExecutionCostModel(changingTokenCheckCost{}); err != nil {
		t.Fatal(err)
	}
	r := serviceRequest("a", 0)
	s.InjectArrival(r)
	drainService(t, s)
	if r.State != StateCompleted || (*records)[0].ExecutionWork.TokenChecks[0].Request != "a" || (*records)[0].ExecutionWork.TokenChecks[0].Tokens == 999 {
		t.Fatal("cost model rewrote trusted runtime work", *records)
	}
}
