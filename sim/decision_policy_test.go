package sim

import (
	"reflect"
	"testing"
)

type testDecisionPolicy struct {
	decide   func(DecisionView) DecisionPlan
	feedback []DecisionFeedback
}

func (p *testDecisionPolicy) Decide(v DecisionView) DecisionPlan { return p.decide(v) }
func (p *testDecisionPolicy) Observe(f DecisionFeedback)         { p.feedback = append(p.feedback, f) }

func decisionFixture() *Simulator {
	s := &Simulator{WaitQ: &WaitQueue{}, RunningBatch: &Batch{}, KVCache: MustNewKVCacheState(16, 2), batchFormation: NewBatchFormation(""),
		maxNumSeqs: 2, maxNumBatchedTokens: 8, longPrefillTokenThreshold: 8, reqNumComputedTokens: map[string]int64{}}
	s.WaitQ.Enqueue(&Request{ID: "a", InputTokens: []TokenID{1, 2, 3, 4, 5}, OutputTokens: make([]TokenID, 50), MaxOutputLen: 100, State: StateQueued})
	s.WaitQ.Enqueue(&Request{ID: "b", InputTokens: []TokenID{6, 7, 8}, OutputTokens: make([]TokenID, 2), State: StateQueued})
	return s
}

func TestDecisionViewCannotObserveFutureOutputsOrArrivals(t *testing.T) {
	s := decisionFixture()
	a := s.decisionView(10)
	s.WaitQ.Peek().OutputTokens = make([]TokenID, 9999)
	// Future arrivals belong to the event queue, not the policy's current queue.
	s.Schedule(&ArrivalEvent{time: 1000, Request: &Request{ID: "future", InputTokens: []TokenID{1, 2, 3}, OutputTokens: []TokenID{9}}})
	b := s.decisionView(10)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("view depends on future output or arrivals")
	}
	if a.Waiting[0].ClientOutputLimit != 100 || a.Waiting[1].ClientOutputLimit != 0 {
		t.Fatal("client budget was inferred from actual output")
	}
	a.Waiting[0].ID = "mutated"
	if s.WaitQ.Peek().ID != "a" {
		t.Fatal("snapshot aliases runtime request")
	}
}

func TestDecisionRejectsEntirePlanBeforeMutation(t *testing.T) {
	cases := map[string]func(DecisionView) DecisionPlan{
		"stale": func(v DecisionView) DecisionPlan { return DecisionPlan{Version: v.Version + 1} },
		"unknown": func(v DecisionView) DecisionPlan {
			return DecisionPlan{Version: v.Version, QueueOrder: []string{"a", "future"}}
		},
		"duplicate": func(v DecisionView) DecisionPlan {
			return DecisionPlan{Version: v.Version, QueueOrder: []string{"a", "a"}}
		},
		"empty": func(v DecisionView) DecisionPlan { return DecisionPlan{Version: v.Version, QueueOrder: []string{}} },
		"zero_cap": func(v DecisionView) DecisionPlan {
			return DecisionPlan{Version: v.Version, QueueOrder: []string{"b", "a"}, TokenCaps: []DecisionTokenCap{{"a", 0}}}
		},
		"negative_cap": func(v DecisionView) DecisionPlan {
			return DecisionPlan{Version: v.Version, TokenCaps: []DecisionTokenCap{{"a", -1}}}
		},
		"large_cap": func(v DecisionView) DecisionPlan {
			return DecisionPlan{Version: v.Version, TokenCaps: []DecisionTokenCap{{"a", 9}}}
		},
		"unknown_cap": func(v DecisionView) DecisionPlan {
			return DecisionPlan{Version: v.Version, TokenCaps: []DecisionTokenCap{{"future", 1}}}
		},
		"repeated_cap": func(v DecisionView) DecisionPlan {
			return DecisionPlan{Version: v.Version, TokenCaps: []DecisionTokenCap{{"a", 1}, {"a", 2}}}
		},
		"mutated_view": func(v DecisionView) DecisionPlan {
			v.Waiting[0].ID = "future"
			return DecisionPlan{Version: v.Version, QueueOrder: []string{"future", "b"}}
		},
	}
	for name, decide := range cases {
		t.Run(name, func(t *testing.T) {
			s := decisionFixture()
			p := &testDecisionPolicy{decide: decide}
			s.SetDecisionPolicy(p, nil)
			func() {
				defer func() {
					if recover() == nil {
						t.Error("invalid plan did not fail")
					}
				}()
				s.prepareDecision(10)
			}()
			if s.WaitQ.Peek().ID != "a" || s.KVCache.UsedBlocks() != 0 || s.WaitQ.Peek().ProgressIndex != 0 {
				t.Fatal("rejected plan changed queue/resources/progress")
			}
			if len(p.feedback) != 1 || p.feedback[0].Status != "rejected" {
				t.Fatal("rejection feedback missing")
			}
		})
	}
}

func TestDecisionJointOrderAndCapsAreExecutedByRuntime(t *testing.T) {
	s := decisionFixture()
	p := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
		return DecisionPlan{Version: v.Version, QueueOrder: []string{"b", "a"}, TokenCaps: []DecisionTokenCap{{"a", 2}, {"b", 1}}}
	}}
	var record DecisionRecord
	s.SetDecisionPolicy(p, func(r DecisionRecord) { record = r })
	caps := s.prepareDecision(10)
	r := s.batchFormation.FormBatch(BatchContext{WaitQ: s.WaitQ, RunningBatch: s.RunningBatch, KVCache: s.KVCache, MaxNumBatchedTokens: 8, MaxNumSeqs: 2, ComputedTokens: s.reqNumComputedTokens, TokenLimits: caps})
	s.RunningBatch = r.RunningBatch
	s.finishDecision(r)
	if len(r.NewlyScheduled) != 2 || r.NewlyScheduled[0].Request.ID != "b" || r.NewlyScheduled[0].Request.NumNewTokens != 1 || r.NewlyScheduled[1].Request.NumNewTokens != 2 {
		t.Fatal("joint decision did not reach allocator/batch formation")
	}
	if len(record.Feedback.Grants) != 2 || record.Feedback.Status != "applied" || record.CostCoverage != "host_measured_sim_time_uncalibrated" {
		t.Fatal("actual grants/cost coverage missing")
	}
	for _, req := range r.RunningBatch.Requests {
		if req.TTFTSet || req.ProgressIndex != 0 {
			t.Fatal("policy advanced tokens or recorded a first token")
		}
	}
}
