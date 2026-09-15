package sim

import "testing"

func TestDecisionPromotionValidatesCompletePlan(t *testing.T) {
	v := DecisionView{Version: 3, BlockTokens: 16, MaxBatchTokens: 128,
		Waiting:      []DecisionRequest{{ID: "waiting", InputTokens: 64}, {ID: "resumed", InputTokens: 64, ComputedTokens: 16}},
		Running:      []DecisionRequest{{ID: "running", InputTokens: 64}},
		Capabilities: DecisionCapabilities{Promotions: true, RestoreChoice: true, AdmissionSelection: true}}
	p := DecisionPlan{Version: 3, Promotions: []DecisionPromotion{{Request: "waiting", MaxPrefixBlocks: 3}}, Admission: &DecisionAdmission{}}
	if err := ValidateDecision(v, p); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"future", "running", "resumed"} {
		bad := cloneDecisionPlan(p)
		bad.Promotions[0].Request = id
		if ValidateDecision(v, bad) == nil {
			t.Fatal("accepted ineligible request", id)
		}
	}
	for _, endpoint := range []int64{-1, 0, 4} {
		bad := cloneDecisionPlan(p)
		bad.Promotions[0].MaxPrefixBlocks = endpoint
		if ValidateDecision(v, bad) == nil {
			t.Fatal("accepted invalid/logits-block endpoint", endpoint)
		}
	}
	bad := cloneDecisionPlan(p)
	bad.Promotions = append(bad.Promotions, bad.Promotions[0])
	if ValidateDecision(v, bad) == nil {
		t.Fatal("accepted duplicate promotion")
	}
	bad = cloneDecisionPlan(p)
	bad.Restores = []DecisionRestore{{Request: "waiting", MaxPrefixBlocks: 0}}
	if ValidateDecision(v, bad) == nil {
		t.Fatal("accepted conflicting demand restore update")
	}
	v.Capabilities.Promotions = false
	if ValidateDecision(v, p) == nil {
		t.Fatal("accepted unsupported backend")
	}
}

func TestPromotionPlanAndFeedbackAreDetached(t *testing.T) {
	p := DecisionPlan{Promotions: []DecisionPromotion{{Request: "r", MaxPrefixBlocks: 2}}}
	c := cloneDecisionPlan(p)
	c.Promotions[0].MaxPrefixBlocks = 99
	f := DecisionFeedback{Promotions: []DecisionPromotionOutcome{{Request: "r", StartedBlocks: 2}}}
	fcopy := cloneDecisionFeedback(f)
	fcopy.Promotions[0].StartedBlocks = 99
	if p.Promotions[0].MaxPrefixBlocks != 2 || f.Promotions[0].StartedBlocks != 2 {
		t.Fatal("policy can mutate saved promotion evidence")
	}
}

func TestPrefetchQueueRetriesSupersededPlanAndFallsBackAfterAttempt(t *testing.T) {
	v := DecisionView{Version: 1, BlockTokens: 16, MaxBatchTokens: 128,
		Waiting:      []DecisionRequest{{ID: "r", InputTokens: 65}},
		KV:           DecisionKVState{PrefixStateKnown: true, Requests: []DecisionKVRequest{{ID: "r", RecoverablePrefixBlocks: 4}}},
		Capabilities: DecisionCapabilities{Promotions: true, QueueOrder: true}}
	p, _ := NewPrefetchQueuePolicy("fcfs")
	first := p.Decide(v)
	if len(first.Promotions) != 1 || first.Admission != nil {
		t.Fatal("prefix warming must retain ordinary admission for no-I/O fallback")
	}
	p.Observe(DecisionFeedback{Status: "superseded"})
	if len(p.Decide(v).Promotions) != 1 {
		t.Fatal("superseded plan consumed a promotion attempt")
	}
	p.Observe(DecisionFeedback{Status: "applied", Promotions: []DecisionPromotionOutcome{{Request: "r", BlockedBlocks: 4}}})
	next := p.Decide(v)
	if len(next.Promotions) != 0 || next.Admission != nil || len(next.QueueOrder) != 1 {
		t.Fatal("blocked action did not fall back to normal admission")
	}
	v.Waiting = nil
	p.Decide(v)
	if len(p.attempted) != 0 {
		t.Fatal("completed request left unbounded policy history")
	}
}
