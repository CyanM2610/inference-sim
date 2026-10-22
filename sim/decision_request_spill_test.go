package sim

import "testing"

func TestRequestSpillValidatesCompleteJointPlan(t *testing.T) {
	v := DecisionView{Version: 7, Capabilities: DecisionCapabilities{PrefillPreemption: true, CapacityPreemption: true, RequestSpill: true},
		Running: []DecisionRequest{{ID: "a", State: StateRunning, ComputedTokens: 4, InputTokens: 8, PrefillPreemptible: true}},
		KV:      DecisionKVState{Pools: []DecisionPool{{ID: "dram"}}}}
	p := DecisionPlan{Version: 7, Preemptions: []string{"a"}, PreemptionStorage: []DecisionPreemptionStorage{{Request: "a", Mode: "spill", Pool: "dram"}}}
	if err := ValidateDecision(v, p); err != nil {
		t.Fatal(err)
	}
	conditional := cloneDecisionPlan(p)
	conditional.Preemptions = nil
	conditional.CapacityVictims = &DecisionCapacityVictims{Order: []string{"a"}}
	if err := ValidateDecision(v, conditional); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*DecisionView, *DecisionPlan){
		func(v *DecisionView, p *DecisionPlan) { v.Capabilities.RequestSpill = false },
		func(v *DecisionView, p *DecisionPlan) { p.Preemptions = nil },
		func(v *DecisionView, p *DecisionPlan) { p.PreemptionStorage[0].Request = "other" },
		func(v *DecisionView, p *DecisionPlan) { p.PreemptionStorage[0].Pool = "remote" },
		func(v *DecisionView, p *DecisionPlan) { p.PreemptionStorage[0].Mode = "recompute" },
		func(v *DecisionView, p *DecisionPlan) { p.PreemptionStorage[0].Mode = "unknown" },
		func(v *DecisionView, p *DecisionPlan) {
			p.PreemptionStorage = append(p.PreemptionStorage, p.PreemptionStorage[0])
		},
		func(v *DecisionView, p *DecisionPlan) { v.Running[0].EmittedTokens = 1 },
	} {
		view, plan := cloneDecisionView(v), cloneDecisionPlan(p)
		mutate(&view, &plan)
		if ValidateDecision(view, plan) == nil {
			t.Fatal("accepted unsupported storage action", view, plan)
		}
	}
	copy := cloneDecisionPlan(p)
	copy.PreemptionStorage[0].Pool = "changed"
	f := DecisionFeedback{PreemptionStorage: []RequestSpillOutcome{{Request: "a", Status: "spill_pending"}}}
	feedback := cloneDecisionFeedback(f)
	feedback.PreemptionStorage[0].Status = "changed"
	if p.PreemptionStorage[0].Pool != "dram" || f.PreemptionStorage[0].Status != "spill_pending" {
		t.Fatal("policy can mutate runtime evidence")
	}
}
