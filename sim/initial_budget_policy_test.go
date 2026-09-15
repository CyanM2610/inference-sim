package sim

import (
	"math"
	"reflect"
	"testing"
)

func initialBudgetView() DecisionView {
	v := budgetView()
	v.Waiting = v.Waiting[:1]
	v.KV.Requests = v.KV.Requests[:1]
	v.Estimates.Requests = v.Estimates.Requests[:1]
	v.Waiting[0].TTFTTargetUS = 60
	return v
}

func TestInitialBudgetSurvivesComputePreemptionAndReentry(t *testing.T) {
	v := initialBudgetView()
	p, _ := NewInitialBudgetRestorePolicy(1, 2)
	p.Decide(v) // first visible at 10, initial=60-20=40, remaining=30
	initial := p.RequestBudgets(10)[0]
	if !initial.Known || initial.InitialUS != 40 || initial.RemainingUS != 30 || initial.InitializedUS != 10 || initial.Provenance != "synthetic test" {
		t.Fatal("missing initial estimate evidence", initial)
	}
	r := v.Waiting[0]
	r.State, r.ComputedTokens, r.PrefillPreemptible = StateRunning, 2, true
	v.Waiting, v.Running, v.Estimates.Requests = nil, []DecisionRequest{r}, nil
	v.NowUS = 20
	p.Decide(v)
	if b := p.RequestBudgets(20)[0]; b.RemainingUS != 20 || b.IntrinsicUS != 20 {
		t.Fatal("running progress refunded budget", b)
	}
	// Logical preemption resets progress, not the immutable initial estimate.
	v = initialBudgetView()
	v.NowUS = 30
	v.KV.Requests[0].LocalPrefixBlocks = 1
	v.KV.Requests[0].LocalPrefixTokens = 2
	v.Estimates.Requests[0].Choices = []DecisionRestoreEstimate{
		{MaxPrefixBlocks: 1, ComputeUS: 60, MaxLoadComputeUS: 60},
		{MaxPrefixBlocks: 2, LoadUS: 15, ComputeUS: 5, MaxLoadComputeUS: 5},
	}
	plan := p.Decide(v)
	if err := ValidateDecisionEstimates(v, *v.Estimates); err != nil {
		t.Fatal(err)
	}
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if plan.Restores[0].MaxPrefixBlocks != 1 || p.RequestBudgets(30)[0].RemainingUS != 10 {
		t.Fatal("reentry or faster prediction refunded already spent time", plan)
	}
	old, _ := NewBudgetRestorePolicy(1, 0)
	if old.Decide(v).Restores[0].MaxPrefixBlocks != 2 {
		t.Fatal("counterexample fails to distinguish old per-step budget")
	}
	// Diagnostic results cannot mutate policy history.
	copy := p.RequestBudgets(30)
	copy[0].InitialUS = 999
	if p.RequestBudgets(30)[0].InitialUS != 40 {
		t.Fatal("diagnostic aliases policy state")
	}
}

func TestInitialBudgetUnknownDelayedEstimateAndCleanup(t *testing.T) {
	v := initialBudgetView()
	p, _ := NewInitialBudgetRestorePolicy(1, 0)
	v.KV.Requests[0].TransferPending = true
	v.Estimates.Requests[0].Choices = nil
	v.Estimates.Requests[0].Unavailable = "restore_in_flight"
	if plan := p.Decide(v); len(plan.Restores) != 0 {
		t.Fatal("in-flight range rewritten")
	}
	if b := p.RequestBudgets(10)[0]; b.Known || b.Reason != "restore_in_flight" {
		t.Fatal("unavailable initial estimate fabricated", b)
	}
	v = initialBudgetView()
	v.NowUS = 25
	p.Decide(v)
	if b := p.RequestBudgets(25)[0]; b.InitializedUS != 25 || b.RemainingUS != 15 {
		t.Fatal("delayed initialization reset arrival clock", b)
	}
	v.KV.Requests[0].TransferPending = true
	v.Estimates.Requests[0].Choices = nil
	v.Estimates.Requests[0].Unavailable = "restore_in_flight"
	v.NowUS = 35
	p.Decide(v)
	if b := p.RequestBudgets(35)[0]; !b.Known || b.RemainingUS != 5 {
		t.Fatal("in-flight request stopped consuming its known budget", b)
	}
	v.Waiting = nil
	v.Running = []DecisionRequest{{ID: "unseen-running", State: StateRunning, InputTokens: 10, ComputedTokens: 4, TTFTTargetUS: 100}}
	v.Estimates.Requests = nil
	p.Decide(v)
	if b := p.RequestBudgets(35); len(b) != 1 || b[0].Known || b[0].Reason != "missing_initial_estimate" {
		t.Fatal("absent request retained or running history fabricated", b)
	}
	v.Running = nil
	p.Decide(v)
	if len(p.RequestBudgets(35)) != 0 {
		t.Fatal("completed requests leaked")
	}
}

func TestInitialBudgetRetainsQueuePriorityDuringRestore(t *testing.T) {
	v := budgetView()
	p, _ := NewInitialBudgetRestorePolicy(1, 0)
	p.Decide(v)
	// Tight's budget must remain known while its load is pending, instead of
	// silently moving behind all other requests due to unavailable new costs.
	v.KV.Requests[2].TransferPending = true
	v.Estimates.Requests[2].Choices = nil
	v.Estimates.Requests[2].Unavailable = "restore_in_flight"
	v.NowUS = 20
	plan := p.Decide(v)
	if !reflect.DeepEqual(plan.QueueOrder, []string{"tight", "mid", "wide"}) || len(plan.Restores) != 2 {
		t.Fatal("lost persistent priority or changed pending range", plan)
	}
}

func TestInitialBudgetRejectsInvalidHistoryWithoutChangingLedger(t *testing.T) {
	for _, mutate := range []func(*DecisionView){
		func(v *DecisionView) { v.NowUS = 9 },
		func(v *DecisionView) { v.Waiting[0].TTFTTargetUS++ },
		func(v *DecisionView) { v.Waiting[0].ArrivalUS++ },
		func(v *DecisionView) { v.Waiting[0].InputTokens++ },
		func(v *DecisionView) { v.Waiting = append(v.Waiting, v.Waiting[0]) },
	} {
		p, _ := NewInitialBudgetRestorePolicy(1, 0)
		p.Decide(initialBudgetView())
		before := p.RequestBudgets(10)
		v := initialBudgetView()
		mutate(&v)
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("invalid history accepted")
				}
			}()
			p.Decide(v)
		}()
		if !reflect.DeepEqual(before, p.RequestBudgets(10)) {
			t.Fatal("rejected view corrupted ledger")
		}
	}
}

func TestInitialBudgetNoTargetAndSaturatingRemaining(t *testing.T) {
	v := initialBudgetView()
	v.Waiting[0].TTFTTargetUS = 0
	p, _ := NewInitialBudgetRestorePolicy(1, 0)
	if p.Decide(v).Restores[0].MaxPrefixBlocks != 2 || p.RequestBudgets(10)[0].Reason != "no_ttft_target" {
		t.Fatal("no-target fallback is not explicit minimum service")
	}
	if remainingInitialBudget(RequestBudget{InitialUS: -10}, math.MaxInt64) != math.MinInt64 {
		t.Fatal("negative priority wrapped to positive")
	}
	for _, gamma := range []float64{0, math.NaN(), math.Inf(1)} {
		if _, err := NewInitialBudgetRestorePolicy(gamma, 0); err == nil {
			t.Fatal("invalid guardband accepted")
		}
	}
}
