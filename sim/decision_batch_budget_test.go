package sim

import (
	"reflect"
	"testing"
)

func TestBatchTokenCapBoundsActualGrantsAndResetsAfterOmission(t *testing.T) {
	s, records := serviceFixture(t, nil)
	first := true
	p := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
		plan := DecisionPlan{Version: v.Version}
		if first {
			plan.BatchTokenCap, first = 2, false
		}
		for _, r := range append(append([]DecisionRequest(nil), v.Running...), v.Waiting...) {
			plan.TokenCaps = append(plan.TokenCaps, DecisionTokenCap{Request: r.ID, Tokens: 8})
		}
		return plan
	}}
	if err := s.SetDecisionPolicy(p, func(r DecisionRecord) { *records = append(*records, r) }); err != nil {
		t.Fatal(err)
	}
	if err := s.SetDecisionCostModel(LinearDecisionCost{FixedUS: 5, Provenance: "synthetic batch-cap service"}); err != nil {
		t.Fatal(err)
	}
	a, b := serviceRequest("a", 0), serviceRequest("b", 0)
	a.InputTokens = []TokenID{1, 2, 3, 4, 5}
	b.InputTokens = []TokenID{6, 7, 8, 9, 10}
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if len(*records) < 2 || a.State != StateCompleted || b.State != StateCompleted {
		t.Fatal("batch cap did not preserve completion")
	}
	var afterOmission int64
	for i, d := range *records {
		var granted int64
		for _, g := range d.Feedback.Grants {
			granted += g.Tokens
		}
		if i == 0 && (granted != 2 || d.Feedback.AtUS != 5 || len(d.Plan.TokenCaps) != 2) {
			t.Fatal("overbooked request caps escaped aggregate service-time cap", d)
		}
		if i > 0 {
			afterOmission = max(afterOmission, granted)
		}
	}
	if afterOmission <= 2 || s.maxNumBatchedTokens != 16 {
		t.Fatal("omitted cap remained active or changed configuration", afterOmission, s.maxNumBatchedTokens)
	}
}

func TestBatchTokenCapRejectsInvalidPlanBeforeQueueMutation(t *testing.T) {
	for _, cap := range []int64{-1, 9} {
		s := decisionFixture()
		p := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
			return DecisionPlan{Version: v.Version, BatchTokenCap: cap, QueueOrder: []string{"b", "a"}}
		}}
		s.SetDecisionPolicy(p, nil)
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("invalid batch cap accepted")
				}
			}()
			s.prepareDecision(0)
		}()
		if s.WaitQ.Peek().ID != "a" || s.decision.batchTokenCap != 0 {
			t.Fatal("rejected cap mutated execution state")
		}
	}
	v := DecisionView{MaxBatchTokens: 16, Running: []DecisionRequest{{ID: "a"}, {ID: "b"}}, Capabilities: DecisionCapabilities{BatchTokenCap: true}}
	if ValidateDecision(v, DecisionPlan{BatchTokenCap: 1}) == nil {
		t.Fatal("cap starves existing running requests")
	}
}

func TestReadyBudgetSeparatesExpectedComputeFromLoadingAdmission(t *testing.T) {
	v := balancedView()
	v.Capabilities.BatchTokenCap, v.Capabilities.RestoreDefersCompute = true, true
	before := cloneDecisionView(v)
	p, _ := NewBalancedBatchPolicyWithOptions(100, 128, 2, 0, BalancedBatchOptions{ReadyComputeBudget: true})
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if plan.BatchTokenCap != 128 || !reflect.DeepEqual(plan.Admission.Requests, []string{"a-hot", "c-cold"}) ||
		!reflect.DeepEqual(plan.TokenCaps, []DecisionTokenCap{{"a-hot", 1}, {"c-cold", 128}}) {
		t.Fatal("pending load consumed the expected compute budget", plan)
	}
	if !reflect.DeepEqual(v, before) {
		t.Fatal("ready-budget policy mutated its snapshot")
	}
	v.KV.Requests[0].LocalPrefixBlocks, v.KV.Requests[0].LocalPrefixTokens = 16, 255
	v.KV.Requests[1].TransferPending = true
	if caps := p.Decide(v).TokenCaps; !reflect.DeepEqual(caps, []DecisionTokenCap{{"a-hot", 1}, {"c-cold", 127}}) {
		t.Fatal("ready hit was incorrectly treated as free compute", caps)
	}
	model := LinearDecisionCost{Provenance: "old profile", Limits: &DecisionCostLimits{MaxVisibleRequests: 3, MaxInputTokens: 256}}
	v.KV.TransfersKnown = true
	warned := false
	model.SetProfileWarningObserver(func(parameter string, observed, maximum int64) {
		warned = parameter == "batch_token_cap" && observed == 128 && maximum == 0
	})
	if _, err := model.Estimate(v, plan); err != nil || !warned {
		t.Fatal("batch budget extrapolation rejected or silent", err)
	}
	model.Limits.MaxBatchTokenCap = 128
	if _, err := model.Estimate(v, plan); err != nil {
		t.Fatal(err)
	}
}
