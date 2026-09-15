package sim

import (
	"reflect"
	"testing"
)

type fixedDecisionEstimator struct {
	result DecisionEstimates
	seen   []DecisionView
}

func (m *fixedDecisionEstimator) Estimate(v DecisionView) (DecisionEstimates, error) {
	m.seen = append(m.seen, v)
	return m.result, nil
}

type estimateTestStore struct{ KVStore }

func (s *estimateTestStore) DecisionState(q []DecisionKVQuery) DecisionKVState {
	v := DecisionKVState{PrefixStateKnown: true}
	for _, r := range q {
		v.Requests = append(v.Requests, DecisionKVRequest{ID: r.ID})
	}
	return v
}

func TestDecisionEstimatesDetachedAndFutureBlind(t *testing.T) {
	s := decisionFixture()
	s.KVCache = &estimateTestStore{s.KVCache}
	m := &fixedDecisionEstimator{result: DecisionEstimates{Provenance: "synthetic", Coverage: "uncalibrated", Requests: []DecisionRequestEstimate{{Request: "a", Choices: []DecisionRestoreEstimate{{ComputeUS: 10}}}, {Request: "b", Choices: []DecisionRestoreEstimate{{ComputeUS: 10}}}}}}
	p := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
		if !v.Capabilities.CostEstimates || v.Estimates.Requests[0].Choices[0].ComputeUS != 10 {
			t.Fatal("prediction missing")
		}
		v.Estimates.Requests[0].Choices[0].ComputeUS = 1000
		return DecisionPlan{Version: v.Version}
	}}
	s.SetDecisionPolicy(p, nil)
	if err := s.SetDecisionEstimator(m); err != nil {
		t.Fatal(err)
	}
	s.prepareDecision(10)
	s.WaitQ.Peek().OutputTokens = make([]TokenID, 99999)
	s.Schedule(&ArrivalEvent{time: 1000, Request: &Request{ID: "future", InputTokens: []TokenID{9}}})
	s.prepareDecision(10)
	if !reflect.DeepEqual(m.seen[0], m.seen[1]) {
		t.Fatal("estimator received true future output or arrival")
	}
	m.result.Requests[0].Choices[0].ComputeUS = 2222
	if s.decision.record.View.Estimates.Requests[0].Choices[0].ComputeUS != 10 {
		t.Fatal("retained estimate aliases caller or policy")
	}
}

func TestInvalidEstimateRejectedBeforePolicyOrResources(t *testing.T) {
	s := decisionFixture()
	called := false
	p := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan { called = true; return DecisionPlan{Version: v.Version} }}
	s.SetDecisionPolicy(p, nil)
	s.SetDecisionEstimator(&fixedDecisionEstimator{}) // missing provenance / coverage
	func() {
		defer func() {
			if recover() == nil {
				t.Error("invalid estimate accepted")
			}
		}()
		s.prepareDecision(10)
	}()
	if called || s.KVCache.UsedBlocks() != 0 || len(p.feedback) != 1 || p.feedback[0].Status != "rejected" {
		t.Fatal("invalid predictor produced a partial decision")
	}
}
