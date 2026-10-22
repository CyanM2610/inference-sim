package sim

import (
	"reflect"
	"testing"
)

func budgetView() DecisionView {
	v := DecisionView{Version: 1, NowUS: 10, BlockTokens: 2, MaxBatchTokens: 8, MaxSequences: 3,
		Capabilities: DecisionCapabilities{RestoreChoice: true, CostEstimates: true, QueueOrder: true, TokenCaps: true}, KV: DecisionKVState{PrefixStateKnown: true},
		Estimates: &DecisionEstimates{Provenance: "synthetic test", Coverage: "uncalibrated"}}
	for _, r := range []DecisionRequest{{ID: "wide", InputTokens: 4, TTFTTargetUS: 60}, {ID: "mid", InputTokens: 4, TTFTTargetUS: 45}, {ID: "tight", InputTokens: 4, TTFTTargetUS: 29}} {
		v.Waiting = append(v.Waiting, r)
		v.KV.Requests = append(v.KV.Requests, DecisionKVRequest{ID: r.ID, RecoverablePrefixBlocks: 2, RecoverablePrefixTokens: 3})
		v.Estimates.Requests = append(v.Estimates.Requests, DecisionRequestEstimate{Request: r.ID, Choices: []DecisionRestoreEstimate{{MaxPrefixBlocks: 0, ComputeUS: 100, MaxLoadComputeUS: 120}, {MaxPrefixBlocks: 1, LoadUS: 10, ComputeUS: 70, MaxLoadComputeUS: 90}, {MaxPrefixBlocks: 2, LoadUS: 20, ComputeUS: 10, MaxLoadComputeUS: 30}}})
	}
	return v
}

func TestBudgetRestoreCombinesOrderAndVariableRestore(t *testing.T) {
	v := budgetView()
	p, err := NewBudgetRestorePolicy(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.QueueOrder, []string{"tight", "mid", "wide"}) {
		t.Fatal("not least remaining budget", plan.QueueOrder)
	}
	want := map[string]int64{"tight": 0, "mid": 1, "wide": 2}
	for _, u := range plan.Restores {
		if u.MaxPrefixBlocks != want[u.Request] {
			t.Fatal("wrong budget choice", u)
		}
	}
	if len(plan.Restores) != 3 || len(plan.TokenCaps) != 3 {
		t.Fatal("joint actions missing")
	}
	// Elapsed time consumes the same budget; no output prediction is needed.
	v.NowUS = 30
	for _, u := range p.Decide(v).Restores {
		if u.Request == "wide" && u.MaxPrefixBlocks != 1 {
			t.Fatal("waiting did not reduce affordable loading", u)
		}
	}
	// A more conservative service estimate also reduces available budget.
	guarded, _ := NewBudgetRestorePolicy(2, 0)
	for _, u := range guarded.Decide(budgetView()).Restores {
		if u.Request == "wide" && u.MaxPrefixBlocks != 1 {
			t.Fatal("guardband ignored")
		}
	}
}

func TestBudgetRestoreSlowLoadNoSLOAndInFlight(t *testing.T) {
	v := budgetView()
	p, _ := NewBudgetRestorePolicy(1, 0)
	for i := range v.Waiting {
		v.Waiting[i].TTFTTargetUS = 0
	}
	plan := p.Decide(v)
	for _, u := range plan.Restores {
		if u.MaxPrefixBlocks != 2 {
			t.Fatal("no-SLO fallback not minimum predicted time")
		}
	}
	for i := range v.Estimates.Requests {
		for j := 1; j < len(v.Estimates.Requests[i].Choices); j++ {
			v.Estimates.Requests[i].Choices[j].LoadUS = 1000
		}
	}
	for _, u := range p.Decide(v).Restores {
		if u.MaxPrefixBlocks != 0 {
			t.Fatal("slow hit forced restoration")
		}
	}
	v.KV.Requests[0].TransferPending = true
	v.Estimates.Requests[0].Choices = nil
	v.Estimates.Requests[0].Unavailable = "restore_in_flight"
	for _, u := range p.Decide(v).Restores {
		if u.Request == "wide" {
			t.Fatal("changed pending load range")
		}
	}
}

func TestBudgetOrderSubtractsServiceRatherThanOnlyDeadlineOrLength(t *testing.T) {
	v := budgetView()
	v.Waiting = v.Waiting[:2]
	v.Estimates.Requests = v.Estimates.Requests[:2]
	v.KV.Requests = v.KV.Requests[:2]
	for i := range v.Waiting {
		v.Waiting[i].TTFTTargetUS = 300
	}
	v.Waiting[1].InputTokens = 8
	for i := range v.Estimates.Requests[1].Choices {
		v.Estimates.Requests[1].Choices[i].ComputeUS += 100
		v.Estimates.Requests[1].Choices[i].MaxLoadComputeUS += 100
	}
	p, _ := NewBudgetRestorePolicy(1, 0)
	plan := p.Decide(v)
	if plan.QueueOrder[0] != "mid" {
		t.Fatal("equal deadline must prioritize the larger remaining service requirement", plan.QueueOrder)
	}
}

func TestCompletionBudgetAvoidsInconsistentPartialRestore(t *testing.T) {
	v := budgetView()
	original, _ := NewBudgetRestorePolicy(1, 0)
	joint, _ := NewCompletionBudgetRestorePolicy(1, 0)
	a, b := original.Decide(v), joint.Decide(v)
	// mid's local/partial costs exceed its deadline. Full recovery is faster
	// and fits the predicted completion bound, despite the old copy-only cap.
	if a.Restores[1].MaxPrefixBlocks != 1 || b.Restores[1].MaxPrefixBlocks != 2 {
		t.Fatal("did not resolve inconsistent partial-restore budget", a.Restores, b.Restores)
	}
	// Tight has no feasible option; choose full restoration as best effort,
	// not an assertion of SLO success and not forced slow recomputation.
	if b.Restores[2].MaxPrefixBlocks != 2 {
		t.Fatal("infeasible request lacks fastest best-effort fallback")
	}
	// Partial restoration remains meaningful when its own total fits and the
	// longer transfer does not; it is not replaced with always-full recovery.
	v.Waiting = v.Waiting[:1]
	v.Waiting[0].TTFTTargetUS = 100
	v.Estimates.Requests = v.Estimates.Requests[:1]
	v.Estimates.Requests[0].Choices = []DecisionRestoreEstimate{{MaxPrefixBlocks: 0, ComputeUS: 100, MaxLoadComputeUS: 100}, {MaxPrefixBlocks: 1, LoadUS: 10, ComputeUS: 60, MaxLoadComputeUS: 70}, {MaxPrefixBlocks: 2, LoadUS: 100, ComputeUS: 10, MaxLoadComputeUS: 10}}
	if joint.Decide(v).Restores[0].MaxPrefixBlocks != 1 {
		t.Fatal("completion-bound partial choice ignored")
	}
}
