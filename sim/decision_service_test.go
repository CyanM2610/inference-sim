package sim

import (
	"math"
	"testing"
)

func serviceFixture(t *testing.T, cost DecisionCostModel) (*Simulator, *[]DecisionRecord) {
	t.Helper()
	cfg := SimConfig{Horizon: math.MaxInt64, KVCacheConfig: NewKVCacheConfig(32, 2, 0, 0, 0, 0), BatchConfig: NewBatchConfig(2, 16, 0), BatchCompletionEvents: true, Seed: 42}
	s, err := NewSimulator(cfg, MustNewKVStoreFromConfig(cfg.KVCacheConfig), &fixedStepModel{stepTime: 10})
	if err != nil {
		t.Fatal(err)
	}
	p, _ := NewQueueDecisionPolicy("sjf", 0)
	var records []DecisionRecord
	if err := s.SetDecisionPolicy(p, func(r DecisionRecord) { records = append(records, r) }); err != nil {
		t.Fatal(err)
	}
	if cost != nil {
		if err := s.SetDecisionCostModel(cost); err != nil {
			t.Fatal(err)
		}
	}
	return s, &records
}

func serviceRequest(id string, at int64) *Request {
	return &Request{ID: id, ArrivalTime: at, InputTokens: []TokenID{TokenID(at + 1), 2, 3}, OutputTokens: []TokenID{9}, State: StateQueued}
}

func drainService(t *testing.T, s *Simulator) {
	t.Helper()
	for n := 0; s.HasPendingEvents(); n++ {
		if n > 1000 {
			t.Fatal("decision service failed to drain")
		}
		s.ProcessNextEvent()
	}
}

func TestDecisionServiceFreezesArrivalsAndDelaysActualWork(t *testing.T) {
	for _, arrival := range []int64{5, 10} {
		s, records := serviceFixture(t, LinearDecisionCost{FixedUS: 10, Provenance: "synthetic test"})
		a, b := serviceRequest("a", 0), serviceRequest("b", arrival)
		s.InjectArrival(a)
		s.InjectArrival(b)
		// A redundant wake must not launch a second concurrent decision.
		s.Schedule(&StepEvent{time: 3})
		for s.PeekNextEventTime() < 10 {
			s.ProcessNextEvent()
			if s.KVCache.UsedBlocks() != 0 || a.ProgressIndex != 0 || a.TTFTSet {
				t.Fatal("resource or computation committed before service finished")
			}
		}
		drainService(t, s)
		if a.State != StateCompleted || b.State != StateCompleted || len(*records) != 2 {
			t.Fatalf("did not complete two distinct plans: a=%s b=%s records=%d", a.State, b.State, len(*records))
		}
		first, second := (*records)[0], (*records)[1]
		if len(first.Feedback.Grants) != 1 || first.Feedback.Grants[0].Request != "a" || len(first.View.Waiting) != 1 || second.View.Waiting[0].ID != "b" {
			t.Fatal("late arrival joined frozen plan or was stranded")
		}
		if first.Feedback.AtUS != 10 || second.Feedback.AtUS != 30 || s.Metrics.RequestTTFTs["a"] != 20 || s.Metrics.RequestTTFTs["b"] != float64(40-arrival) {
			t.Fatalf("service not paid on execution path: records=%+v ttft=%v", *records, s.Metrics.RequestTTFTs)
		}
		if first.Service.ExtraUS != 10 || first.Service.EndUS-first.Service.StartUS != 10 || first.CostCoverage != ExplicitDecisionCostCoverage {
			t.Fatal("explicit cost evidence missing")
		}
	}
}

func TestDecisionServiceTimeoutSupersedesPlanWithoutRefund(t *testing.T) {
	s, records := serviceFixture(t, LinearDecisionCost{FixedUS: 10, Provenance: "synthetic test"})
	a, b := serviceRequest("a", 0), serviceRequest("b", 0)
	a.Deadline = 5
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if a.State != StateTimedOut || b.State != StateCompleted || len(*records) != 2 {
		t.Fatalf("wrong terminal state or plan count: %s %s %d", a.State, b.State, len(*records))
	}
	first, second := (*records)[0], (*records)[1]
	if first.Feedback.Status != "superseded" || len(first.Feedback.Grants) != 0 || first.Feedback.AtUS != 10 || second.View.NowUS != 10 || second.Feedback.AtUS != 20 || s.Metrics.RequestTTFTs["b"] != 30 {
		t.Fatalf("superseded service was applied/refunded: %+v", *records)
	}
	if len(second.View.Waiting) != 1 || second.View.Waiting[0].ID != "b" || s.stepEvent != nil || s.decision.pending != nil {
		t.Fatal("stale membership or in-flight marker remains")
	}
}

func TestDecisionServiceRunningTimeoutDrainsLastPlan(t *testing.T) {
	s, records := serviceFixture(t, LinearDecisionCost{FixedUS: 10, Provenance: "synthetic test"})
	a := serviceRequest("a", 0)
	a.OutputTokens = []TokenID{8, 9}
	a.Deadline = 25 // first output at 20; second decision service occupies [20,30]
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateTimedOut || len(*records) != 2 || (*records)[1].Feedback.Status != "superseded" || len((*records)[1].View.Running) != 1 {
		t.Fatalf("running timeout did not invalidate paid plan: %+v", *records)
	}
	if s.RunningBatch != nil || s.stepEvent != nil || s.decision.pending != nil || s.Clock != 30 {
		t.Fatal("last running timeout left phantom work or discarded paid time")
	}
}

type fixedDecisionEstimate struct{ DecisionCostEstimate }

func (c fixedDecisionEstimate) Estimate(DecisionView, DecisionPlan) (DecisionCostEstimate, error) {
	return c.DecisionCostEstimate, nil
}

func TestDecisionServiceRejectsInvalidCostsBeforeResourceMutation(t *testing.T) {
	for name, tc := range map[string]struct {
		model DecisionCostModel
		now   int64
	}{
		"negative":             {fixedDecisionEstimate{DecisionCostEstimate{-1, "test"}}, 0},
		"no_provenance":        {fixedDecisionEstimate{DecisionCostEstimate{1, " "}}, 0},
		"time_overflow":        {fixedDecisionEstimate{DecisionCostEstimate{10, "test"}}, math.MaxInt64 - 5},
		"coefficient_overflow": {LinearDecisionCost{FixedUS: 1, PerVisibleRequestUS: math.MaxInt64, Provenance: "test"}, 0},
	} {
		t.Run(name, func(t *testing.T) {
			s, records := serviceFixture(t, tc.model)
			s.WaitQ.Enqueue(serviceRequest("a", 0))
			func() {
				defer func() {
					if recover() == nil {
						t.Error("invalid cost accepted")
					}
				}()
				s.Step(tc.now)
			}()
			if s.HasPendingEvents() || s.KVCache.UsedBlocks() != 0 || s.WaitQ.Len() != 1 || s.decision.pending != nil {
				t.Fatal("invalid cost left resources or events")
			}
			if len(*records) != 1 || (*records)[0].Feedback.Status != "rejected" {
				t.Fatal("missing rejection evidence")
			}
		})
	}
}

func TestDecisionServiceLinearCostUsesOnlyVisibleCounts(t *testing.T) {
	c := LinearDecisionCost{FixedUS: 2, PerVisibleRequestUS: 3, PerPendingTransferUS: 5, Provenance: "test"}
	v := DecisionView{Waiting: make([]DecisionRequest, 2), Running: make([]DecisionRequest, 1), KV: DecisionKVState{TransfersKnown: true, PendingTransfers: make([]DecisionTransfer, 2)}}
	e, err := c.Estimate(v, DecisionPlan{})
	if err != nil || e.ExtraUS != 21 {
		t.Fatal(e, err)
	}
	v.KV.TransfersKnown = false
	if _, err := c.Estimate(v, DecisionPlan{}); err == nil {
		t.Fatal("unknown transfers were charged as a known count")
	}
}

type serviceBlockedStore struct{ KVStore }

func (k serviceBlockedStore) AllocateKVBlocks(r *Request, start, end int64, cached []int64) bool {
	return r.ID != "blocked" && k.KVStore.AllocateKVBlocks(r, start, end, cached)
}

func TestDecisionServiceLateArrivalWakesAfterBlockedFrozenBatch(t *testing.T) {
	s, records := serviceFixture(t, LinearDecisionCost{FixedUS: 10, Provenance: "synthetic test"})
	s.KVCache = serviceBlockedStore{s.KVCache}
	a, b := serviceRequest("blocked", 0), serviceRequest("b", 5)
	a.InputTokens = append(a.InputTokens, 4, 5) // SJF selects b on the next decision.
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if b.State != StateCompleted || len(*records) < 2 || (*records)[1].View.NowUS != 10 || (*records)[1].Feedback.Grants[0].Request != "b" {
		t.Fatal("late arrival stranded after the old plan blocked without I/O")
	}
	if a.State != StateQueued || s.WaitQ.Len() != 1 {
		t.Fatal("fixture's explicitly blocked request was lost")
	}
}
