package sim

import "testing"

func TestPrefillDeferralLastsOneDecisionAndPreservesKV(t *testing.T) {
	s, policy, records := waitFixture(t, 20, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 2 {
			p.PrefillDeferrals = []string{"long"}
			p.BatchTokenCap = 1
		}
		return p
	})
	s.longPrefillTokenThreshold, s.maxNumBatchedTokens = 2, 3
	if err := s.EnableDecisionPrefillDeferrals(); err != nil {
		t.Fatal(err)
	}
	long, decode := longWaitRequest(), serviceRequest("decode", 0)
	decode.InputTokens = []TokenID{100}
	decode.OutputTokens = []TokenID{7, 8, 9}
	s.InjectArrival(long)
	s.InjectArrival(decode)
	drainService(t, s)
	if long.State != StateCompleted || decode.State != StateCompleted || s.KVCache.UsedBlocks() != 0 || s.Metrics.PreemptionCount != 0 {
		t.Fatal("deferral lost state or resources")
	}
	if len(timerEvents(policy)) != 0 {
		t.Fatal("one-round deferral fabricated a deadline")
	}
	var held, resumed bool
	var work int64
	for _, d := range *records {
		for _, g := range d.Feedback.Grants {
			if g.Request == "long" {
				work += g.Tokens
				if d.View.Version > 2 {
					resumed = true
				}
				if d.View.Version == 2 {
					t.Fatal("deferred prefill computed after CPU service", d)
				}
			}
		}
		if d.View.Version == 2 {
			held = len(d.Feedback.Grants) == 1 && d.Feedback.Grants[0].Request == "decode"
			if !d.View.Running[0].PrefillDeferrable || d.Feedback.AtUS-d.View.NowUS != 20 {
				t.Fatal("did not exercise deferral across paid service", d)
			}
			if len(d.Feedback.Unselected) != 1 || d.Feedback.Unselected[0].Reason != "policy_prefill_deferred" {
				t.Fatal("missing actual deferral feedback", d)
			}
		}
	}
	if !held || !resumed || work != 6 {
		t.Fatal("deferral did not preserve and resume prefill", held, resumed, work)
	}
}

func TestPrefillDeferralValidationAndWakeup(t *testing.T) {
	v := DecisionView{Version: 1, NowUS: 10, MaxBatchTokens: 4,
		Capabilities: DecisionCapabilities{PrefillDeferrals: true, BatchTokenCap: true, Revisit: true},
		Running:      []DecisionRequest{{ID: "p", State: StateRunning, ComputedTokens: 2, InputTokens: 8, PrefillDeferrable: true}, {ID: "d", State: StateRunning, InputTokens: 1, ComputedTokens: 2, EmittedTokens: 1}}}
	valid := DecisionPlan{Version: 1, PrefillDeferrals: []string{"p"}, BatchTokenCap: 1}
	if err := ValidateDecision(v, valid); err != nil {
		t.Fatal(err)
	}
	for _, ids := range [][]string{{"d"}, {"missing"}, {"p", "p"}} {
		p := valid
		p.PrefillDeferrals = ids
		if ValidateDecision(v, p) == nil {
			t.Fatal("accepted invalid deferral", ids)
		}
	}
	v.Running = v.Running[:1]
	if ValidateDecision(v, valid) == nil {
		t.Fatal("accepted a no-progress plan without a wakeup")
	}
	valid.RevisitAtUS = atUS(30)
	if err := ValidateDecision(v, valid); err != nil {
		t.Fatal("explicit revisit should support all-held prefill", err)
	}
	v.Capabilities.PrefillDeferrals = false
	if ValidateDecision(v, valid) == nil {
		t.Fatal("accepted unsupported action")
	}
}

func TestPrefillDeferralRevisitRunsNoEmptyGPUWork(t *testing.T) {
	s, _, records := waitFixture(t, 0, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 2 {
			p.PrefillDeferrals = []string{"long"}
			p.RevisitAtUS = atUS(30)
		}
		return p
	})
	s.longPrefillTokenThreshold = 2
	if err := s.EnableDecisionPrefillDeferrals(); err != nil {
		t.Fatal(err)
	}
	r := longWaitRequest()
	s.InjectArrival(r)
	drainService(t, s)
	if r.State != StateCompleted || r.FirstTokenTime != 50 || len(*records) != 4 || s.KVCache.UsedBlocks() != 0 {
		t.Fatal("lost wakeup or generated fake compute", r.FirstTokenTime, len(*records))
	}
	if len((*records)[1].Feedback.Grants) != 0 {
		t.Fatal("empty round granted compute")
	}
}

func TestPrefillDeferralExpiredWakeupReplansAfterPaidService(t *testing.T) {
	s, _, records := waitFixture(t, 20, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 2 {
			p.PrefillDeferrals = []string{"long"}
			p.RevisitAtUS = atUS(v.NowUS + 5)
		}
		return p
	})
	s.longPrefillTokenThreshold = 2
	if err := s.EnableDecisionPrefillDeferrals(); err != nil {
		t.Fatal(err)
	}
	r := longWaitRequest()
	s.InjectArrival(r)
	drainService(t, s)
	if r.State != StateCompleted || s.KVCache.UsedBlocks() != 0 {
		t.Fatal("expired wakeup stranded request")
	}
	d := (*records)[1]
	if d.Feedback.Status != "superseded" || d.Feedback.Reason != "prefill_deferral_state_changed" || d.Service.ExtraUS != 20 || len(d.Feedback.Grants) != 0 {
		t.Fatal("expired hold mutated resources or refunded CPU service", d)
	}
}

func TestPrefillDeferralRevisitDuringHostServiceIsNotLost(t *testing.T) {
	s, store, _ := hostFixture(t, 0)
	s.longPrefillTokenThreshold = 2
	policy := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 2 {
			p.PrefillDeferrals = []string{"long"}
			p.RevisitAtUS = atUS(v.NowUS + 7)
		}
		return p
	}}
	if err := s.SetDecisionPolicy(policy, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableDecisionPrefillDeferrals(); err != nil {
		t.Fatal(err)
	}
	r := longWaitRequest()
	s.InjectArrival(r)
	drainService(t, s)
	if r.State != StateCompleted || len(store.starts) != 3 || s.KVCache.UsedBlocks() != 0 || s.stepEvent != nil {
		t.Fatal("host service lost the resident deferral revisit or created an empty GPU step", r.State, store.starts)
	}
}
