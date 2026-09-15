package sim

import "testing"

func TestDeferralProbeCountsOnlyActualHolds(t *testing.T) {
	base := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan { return DecisionPlan{Version: v.Version} }}
	p, err := NewPrefillDeferralProbe(base, 1, 2, []string{"p"})
	if err != nil {
		t.Fatal(err)
	}
	v := DecisionView{Capabilities: DecisionCapabilities{PrefillDeferrals: true}, Running: []DecisionRequest{
		{ID: "p", State: StateRunning, InputTokens: 8, ComputedTokens: 2, PrefillDeferrable: true},
		{ID: "d", State: StateRunning, InputTokens: 1, ComputedTokens: 2, EmittedTokens: 1}}}
	for i, reason := range []string{"preempted", "policy_prefill_deferred"} {
		v.Version = uint64(i + 1)
		plan := p.Decide(v)
		if len(plan.PrefillDeferrals) != 1 {
			t.Fatal("planned attempt consumed an unexecuted hold")
		}
		p.Observe(DecisionFeedback{Version: v.Version, Status: "applied", Unselected: []DecisionUnselected{{Request: "p", Reason: reason}}})
	}
	v.Version++
	if len(p.Decide(v).PrefillDeferrals) != 0 {
		t.Fatal("actual hold budget was not exhausted")
	}
	p.(interface{ Reset() }).Reset()
	v.Running[1].EmittedTokens = 0
	if len(p.Decide(v).PrefillDeferrals) != 0 {
		t.Fatal("probe used a prefill as a decode anchor")
	}
}
