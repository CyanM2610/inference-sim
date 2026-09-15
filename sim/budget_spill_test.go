package sim

import (
	"math"
	"testing"
)

func budgetSpillView() DecisionView {
	v := initialBudgetView()
	v.Capabilities.RequestSpill, v.Capabilities.CapacityPreemption, v.Capabilities.PrefillPreemption = true, true, true
	v.KV.Pools = []DecisionPool{{ID: "dram", CapacityBlocks: 4, FreeBlocks: 4}}
	v.Estimates.SpillProvenance, v.Estimates.SpillCoverage = "synthetic spill", "uncalibrated"
	return v
}

func budgetSpillRunning(v DecisionView, now, cost int64) DecisionView {
	r := v.Waiting[0]
	r.State, r.ComputedTokens, r.PrefillPreemptible = StateRunning, 2, true
	v.Waiting, v.Running, v.Estimates.Requests = nil, []DecisionRequest{r}, nil
	v.Version, v.NowUS = v.Version+1, now
	v.KV.Requests[0].SpillTargets = []DecisionSpillTarget{{Pool: "dram", CompleteBlocks: 1, MissingBlocks: 1, AvailableBlocks: 4}}
	v.Estimates.Spills = []DecisionSpillEstimate{{Request: r.ID, Pool: "dram", CapacityPossible: true, StoreUS: cost, CopyUS: cost}}
	return v
}

func TestBudgetSpillUsesRemainingBudgetAndPreservesRequeueHistory(t *testing.T) {
	base, _ := NewBudgetPressurePolicy(1, 0)
	p, err := NewBudgetPreemptionStoragePolicy(base, "dram", 1)
	if err != nil {
		t.Fatal(err)
	}
	v := budgetSpillView()
	p.Decide(v)
	v = budgetSpillRunning(v, 20, 20)
	if plan := p.Decide(v); len(plan.PreemptionStorage) != 1 || plan.PreemptionStorage[0].Mode != "spill" {
		t.Fatal("exact remaining budget should fit", plan)
	}
	if b := base.RequestBudgets(20)[0]; b.InitialUS != 40 || b.RemainingUS != 20 {
		t.Fatal("wrong retained budget", b)
	}
	// Requeue with a faster new prediction must not refund elapsed time.
	requeued := budgetSpillView()
	requeued.Version, requeued.NowUS = 3, 30
	requeued.Estimates.Requests[0].Choices[2].ComputeUS = 1
	p.Decide(requeued)
	v = budgetSpillRunning(requeued, 31, 20)
	if plan := p.Decide(v); plan.PreemptionStorage[0].Mode != "recompute" {
		t.Fatal("requeue reset consumed budget", plan)
	}
	if b := base.RequestBudgets(31)[0]; b.InitialUS != 40 || b.RemainingUS != 9 || b.ArrivalUS != 0 {
		t.Fatal("history changed on spill/requeue", b)
	}
}

func TestBudgetSpillGuardsCapacityUnknownAndIntegerBoundary(t *testing.T) {
	for _, tc := range []struct {
		name   string
		guard  float64
		cost   int64
		mutate func(*DecisionView)
		want   string
	}{
		{"over-budget", 1, 21, nil, "recompute"},
		{"guarded-up", 1.01, 20, nil, "recompute"},
		{"guarded-fit", 1.25, 16, nil, "spill"},
		{"capacity", 1, 1, func(v *DecisionView) {
			v.KV.Requests[0].SpillTargets[0].AvailableBlocks = 0
			v.Estimates.Spills[0].CapacityPossible = false
		}, "recompute"},
		{"unknown", 1, 0, func(v *DecisionView) {
			v.KV.Requests[0].SpillTargets = nil
			v.Estimates.Spills[0] = DecisionSpillEstimate{Request: "wide", Pool: "dram", Unavailable: "missing_spill_inventory"}
		}, "recompute"},
		{"already-ready", 1, 0, func(v *DecisionView) {
			v.KV.Requests[0].SpillTargets[0].MissingBlocks = 0
			v.KV.Requests[0].SpillTargets[0].ReadyBlocks = 1
		}, "spill"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base, _ := NewBudgetPressurePolicy(1, 0)
			p, _ := NewBudgetPreemptionStoragePolicy(base, "dram", tc.guard)
			v := budgetSpillView()
			p.Decide(v)
			v = budgetSpillRunning(v, 20, tc.cost)
			if tc.mutate != nil {
				tc.mutate(&v)
			}
			plan := p.Decide(v)
			if len(plan.PreemptionStorage) != 1 || plan.PreemptionStorage[0].Mode != tc.want {
				t.Fatal(plan)
			}
		})
	}
	base, _ := NewBudgetPressurePolicy(1, 0)
	for _, guard := range []float64{0, -1, .99, math.Inf(1), math.NaN()} {
		if _, err := NewBudgetPreemptionStoragePolicy(base, "dram", guard); err == nil {
			t.Fatal("invalid guardband accepted")
		}
	}
}

func TestBudgetSpillRejectsMalformedPredictionBeforeBudgetMutation(t *testing.T) {
	base, _ := NewBudgetPressurePolicy(1, 0)
	p, _ := NewBudgetPreemptionStoragePolicy(base, "dram", 1)
	v := budgetSpillView()
	p.Decide(v)
	v = budgetSpillRunning(v, 20, 20)
	v.Estimates.Spills[0].StoreUS = -1
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("invalid prediction accepted")
			}
		}()
		p.Decide(v)
	}()
	if base.budgetNowUS != 10 {
		t.Fatal("invalid spill estimate changed base history")
	}
	cloned := cloneDecisionView(v)
	cloned.KV.Requests[0].SpillTargets[0].MissingBlocks = 77
	cloned.Estimates.Spills[0].StoreUS = 888
	if v.KV.Requests[0].SpillTargets[0].MissingBlocks != 1 || v.Estimates.Spills[0].StoreUS != -1 {
		t.Fatal("spill state/prediction alias runtime")
	}
}
