package sim

import "testing"

func reestimateView(now, cost int64, decodes bool) DecisionView {
	v := budgetQueueView()
	v.Version, v.NowUS, v.MaxSequences = uint64(now+1), now, 3
	v.Waiting[0].TTFTTargetUS, v.Waiting[1].TTFTTargetUS = 200, 100
	v.Estimates.Requests[0].Choices[0].ComputeUS, v.Estimates.Requests[0].Choices[0].MaxLoadComputeUS = 10, 10
	v.Estimates.Requests[1].Choices[0].ComputeUS, v.Estimates.Requests[1].Choices[0].MaxLoadComputeUS = cost, cost
	if decodes {
		v.Running = []DecisionRequest{{ID: "decoder", State: StateRunning, InputTokens: 16, ComputedTokens: 16, EmittedTokens: 1, ClientOutputLimit: 2}}
		v.KV.Requests = append(v.KV.Requests, DecisionKVRequest{ID: "decoder"})
	}
	return v
}

func refreshPolicy() *BudgetQueuePolicy {
	p, _ := NewBudgetQueuePolicyWithOptions(1, 0, 4, 1, BudgetQueueOptions{ReestimateOnDecodeDrop: true})
	return p
}

func bestBudget(p *BudgetQueuePolicy, now int64) RequestBudget {
	for _, b := range p.RequestBudgets(now) {
		if b.Request == "best" {
			return b
		}
	}
	panic("missing best budget")
}

func decideRefresh(p *BudgetQueuePolicy, v DecisionView, grants ...DecisionGrant) DecisionPlan {
	plan := p.Decide(v)
	p.Observe(DecisionFeedback{Version: v.Version, AtUS: v.NowUS, Status: "applied", Grants: grants})
	return plan
}

func TestBudgetReestimatePromotesAfterDecodeDropWithoutResettingArrival(t *testing.T) {
	p, old := refreshPolicy(), func() *BudgetQueuePolicy { q, _ := NewBudgetQueuePolicy(1, 0, 4, 1); return q }()
	for _, policy := range []*BudgetQueuePolicy{p, old} {
		decideRefresh(policy, reestimateView(0, 140, true), DecisionGrant{Request: "best", Tokens: 1})
	}
	v := reestimateView(10, 20, false)
	plan := p.Decide(v)
	baseline := old.Decide(v)
	b := bestBudget(p, 10)
	if b.InitialUS != -40 || b.IntrinsicUS != 140 || b.InitializedUS != 0 || b.ReestimateCount != 1 || b.ReestimatedUS != 10 || b.ReestimatedBudgetUS != 80 || b.RemainingUS != 70 {
		t.Fatal("reestimate lost initial history or elapsed waiting", b)
	}
	if plan.QueueOrder[0] != "best" || capsByRequest(plan)["best"] <= capsByRequest(baseline)["best"] || bestBudget(old, 10).RemainingUS != -50 {
		t.Fatal("refresh did not change real proposed quota/order", plan, baseline)
	}
	p.Observe(DecisionFeedback{Version: v.Version, Status: "applied", Grants: []DecisionGrant{{Request: "best", Tokens: 7}}})
	if s := p.ServiceState(); s.BestEffortTokens != 1 || s.PrimaryTokens != 7 {
		t.Fatal("promotion retroactively reclassified old actual grants", s)
	}
}

func TestBudgetReestimateDoesNotRefreshOnPrefixGrowthOrAnUnchangedLoad(t *testing.T) {
	for _, decodes := range []bool{false, true} {
		p := refreshPolicy()
		decideRefresh(p, reestimateView(0, 140, decodes))
		v := reestimateView(10, 20, decodes)
		v.KV.Requests[1].LocalPrefixBlocks, v.KV.Requests[1].LocalPrefixTokens = 1, 16
		decideRefresh(p, v)
		if b := bestBudget(p, 10); b.ReestimateCount != 0 || b.ReestimatePending || b.RemainingUS != -50 {
			t.Fatal("estimate or prefix change manufactured a load drop", b)
		}
	}
}

func TestBudgetReestimateDefersBlockedLoadAndCancelsOnRebound(t *testing.T) {
	for _, rebound := range []bool{false, true} {
		p := refreshPolicy()
		decideRefresh(p, reestimateView(0, 140, true))
		v := reestimateView(10, 20, false)
		v.KV.Requests[1].TransferPending, v.KV.Requests[1].Deferred = true, true
		v.Estimates.Requests[1].Choices, v.Estimates.Requests[1].Unavailable = nil, "restore_in_flight"
		decideRefresh(p, v)
		if b := bestBudget(p, 10); !b.ReestimatePending || b.ReestimateCount != 0 {
			t.Fatal("lost blocked refresh opportunity", b)
		}
		decideRefresh(p, reestimateView(20, 20, rebound))
		b := bestBudget(p, 20)
		if b.ReestimatePending || (!rebound && (b.ReestimateCount != 1 || b.RemainingUS != 60)) || (rebound && b.ReestimateCount != 0) {
			t.Fatal("wrong blocked/rebound transition", rebound, b)
		}
	}
}

func TestBudgetReestimateCannotReviveAnExpiredDeadlineAndResetClearsHistory(t *testing.T) {
	p := refreshPolicy()
	decideRefresh(p, reestimateView(0, 140, true))
	decideRefresh(p, reestimateView(110, 0, false))
	if b := bestBudget(p, 110); b.ReestimateCount != 1 || b.RemainingUS != -10 {
		t.Fatal("expired deadline revived", b)
	}
	p.Reset()
	decideRefresh(p, reestimateView(0, 20, false))
	if b := bestBudget(p, 0); b.ReestimateCount != 0 || b.InitialUS != 80 || b.ReestimatePending {
		t.Fatal("new workload inherited a refresh", b)
	}
}

func TestBudgetReestimateRejectedIdentityCannotConsumeTheLoadDrop(t *testing.T) {
	p := refreshPolicy()
	decideRefresh(p, reestimateView(0, 140, true))
	v := reestimateView(10, 20, false)
	v.Waiting[1].ArrivalUS = 1
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("changed request identity accepted")
			}
		}()
		p.Decide(v)
	}()
	decideRefresh(p, reestimateView(10, 20, false))
	if b := bestBudget(p, 10); b.ReestimateCount != 1 {
		t.Fatal("invalid view consumed the load drop", b)
	}
}
