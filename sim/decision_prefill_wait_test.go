package sim

import (
	"reflect"
	"strings"
	"testing"
)

func residentWaitFixture(t *testing.T, cost int64, decide func(DecisionView) DecisionPlan) (*Simulator, *timerTestPolicy, *[]DecisionRecord) {
	t.Helper()
	s, p, records := waitFixture(t, cost, decide)
	s.longPrefillTokenThreshold = 2
	if err := s.EnableDecisionPrefillWaits(); err != nil {
		t.Fatal(err)
	}
	return s, p, records
}

func longWaitRequest() *Request {
	r := serviceRequest("long", 0)
	r.InputTokens = []TokenID{1, 2, 3, 4, 5, 6}
	return r
}

func pauseLong(v DecisionView) DecisionPlan {
	p := DecisionPlan{Version: v.Version}
	if v.Version == 2 {
		p.Waits = []DecisionWait{{Request: "long", UntilUS: 40}}
	}
	return p
}

func TestResidentPrefillWaitRetainsProgressKVAndWakesWithoutFakeCompute(t *testing.T) {
	s, p, records := residentWaitFixture(t, 0, pauseLong)
	r := longWaitRequest()
	s.InjectArrival(r)
	batches := 0
	for n := 0; s.HasPendingEvents(); n++ {
		if n > 30 {
			t.Fatal("resident wait busy-looped")
		}
		e := s.ProcessNextEvent()
		if b, ok := e.(*BatchCompleteEvent); ok {
			batches++
			if len(b.Requests) != 1 || b.Requests[0] != "long" || b.Start >= 10 && b.Start < 40 {
				t.Fatal("wait emitted a fake GPU batch", b)
			}
		}
		if s.Clock >= 10 && s.Clock < 40 && (r.ProgressIndex != 2 || r.State != StateRunning || s.KVCache.UsedBlocks() != 1 || s.WaitQ.Len() != 0) {
			t.Fatal("resident wait released KV, changed progress or requeued", r.ProgressIndex, r.State, s.KVCache.UsedBlocks())
		}
	}
	if r.State != StateCompleted || r.FirstTokenTime != 60 || batches != 3 || s.Metrics.PreemptionCount != 0 || len(*records) != 4 {
		t.Fatal("resident wait failed to resume/completed twice", r.State, r.FirstTokenTime, batches, len(*records))
	}
	if !reflect.DeepEqual((*records)[1].Feedback.Unselected, []DecisionUnselected{{Request: "long", Reason: "policy_wait"}}) || len(timerEvents(p)) != 1 || timerEvents(p)[0].Request != "long" {
		t.Fatal("missing actual wait/expiry evidence")
	}
	if s.KVCache.UsedBlocks() != 0 || len(s.decision.waits) != 0 || s.decision.timer != nil || s.stepEvent != nil {
		t.Fatal("resident wait leaked ownership or a timer")
	}
}

func TestResidentPrefillWaitProtectsDecodeAndReleasesTokenBudget(t *testing.T) {
	s, _, records := residentWaitFixture(t, 0, func(v DecisionView) DecisionPlan {
		p := pauseLong(v)
		if v.Version == 2 {
			p.BatchTokenCap = 1 // two residents, but only decode is runnable
		}
		return p
	})
	s.maxNumBatchedTokens = 3
	long, decode := longWaitRequest(), serviceRequest("decode", 0)
	decode.InputTokens = []TokenID{100}
	decode.OutputTokens = []TokenID{7, 8, 9}
	s.InjectArrival(long)
	s.InjectArrival(decode)
	drainService(t, s)
	if long.State != StateCompleted || decode.State != StateCompleted || decode.FirstTokenTime != 10 || s.Metrics.RequestCompletionTimes["decode"] != 30 {
		t.Fatal("resident prefill wait stalled decode", decode.FirstTokenTime, s.Metrics.RequestCompletionTimes)
	}
	for _, d := range *records {
		if d.View.NowUS >= 10 && d.View.NowUS < 40 {
			for _, g := range d.Feedback.Grants {
				if g.Request == "long" {
					t.Fatal("paused prefill consumed tokens", d)
				}
			}
		}
	}
}

func TestResidentPrefillWaitCanClearAndRemoveOldTimer(t *testing.T) {
	s, p, records := residentWaitFixture(t, 0, func(v DecisionView) DecisionPlan {
		p := pauseLong(v)
		if v.Version == 2 {
			p.RevisitAtUS = atUS(20)
		}
		if v.Version == 3 {
			p.Waits = []DecisionWait{{Request: "long", UntilUS: 0}}
		}
		return p
	})
	r := longWaitRequest()
	s.InjectArrival(r)
	drainService(t, s)
	if r.FirstTokenTime != 40 || r.State != StateCompleted || len(timerEvents(p)) != 1 || (*records)[2].Feedback.WaitUpdates[0].Status != "cleared" {
		t.Fatal("explicit clear failed to resume/cancel deadline", r.FirstTokenTime, timerEvents(p))
	}
}

func TestResidentPrefillWaitRetainsSequenceSlot(t *testing.T) {
	s, _, records := residentWaitFixture(t, 0, pauseLong)
	s.maxNumSeqs = 1
	a, b := longWaitRequest(), serviceRequest("short", 15)
	b.InputTokens = []TokenID{20}
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if a.State != StateCompleted || b.State != StateCompleted || s.Metrics.PreemptionCount != 0 {
		t.Fatal("resident wait did not preserve request ownership")
	}
	for _, d := range *records {
		if d.View.NowUS < 60 {
			for _, g := range d.Feedback.Grants {
				if g.Request == "short" {
					t.Fatal("waiting request stole retained sequence slot", d)
				}
			}
		}
	}
}

func TestResidentPrefillWaitElapsedDuringDecisionServiceRunsNormally(t *testing.T) {
	s, p, records := residentWaitFixture(t, 20, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 2 {
			p.Waits = []DecisionWait{{Request: "long", UntilUS: v.NowUS + 5}}
		}
		return p
	})
	r := longWaitRequest()
	s.InjectArrival(r)
	drainService(t, s)
	if r.State != StateCompleted || r.FirstTokenTime != 90 || len(timerEvents(p)) != 0 || (*records)[1].Feedback.WaitUpdates[0].Status != "elapsed" || len((*records)[1].Feedback.Grants) != 1 {
		t.Fatal("elapsed resident wait was applied retroactively", r.FirstTokenTime, *records)
	}
}

func TestResidentPrefillWaitExpiryDuringHostServiceDoesNotLoseWakeup(t *testing.T) {
	s, store, hostRecords := hostFixture(t, 0)
	s.longPrefillTokenThreshold = 2
	policy := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 2 {
			p.Waits = []DecisionWait{{Request: "long", UntilUS: v.NowUS + 7}}
		}
		return p
	}}
	if err := s.SetDecisionPolicy(policy, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableDecisionPrefillWaits(); err != nil {
		t.Fatal(err)
	}
	r := longWaitRequest()
	s.InjectArrival(r)
	drainService(t, s)
	if r.State != StateCompleted || len(store.starts) != 3 || s.KVCache.UsedBlocks() != 0 || s.stepEvent != nil {
		t.Fatal("host completion lost held-request expiry or generated empty GPU work", r.State, store.starts)
	}
	if !reflect.DeepEqual(store.starts, []int64{0, 35, 60}) {
		t.Fatal("did not exercise expiry inside 25..35 host service", store.starts)
	}
	empty := 0
	for _, record := range *hostRecords {
		if record.Work.Stage == "completion" && len(record.Work.Requests) == 0 {
			empty++
			if record.Service.StartUS != 25 || record.Service.EndUS != 35 {
				t.Fatal("empty service did not span wait deadline 32", record)
			}
		}
	}
	if empty != 1 {
		t.Fatal("empty host completion not observed", empty)
	}
}

func TestResidentPrefillWaitExpirySupersedesAnInsufficientBatchCap(t *testing.T) {
	s, _, records := residentWaitFixture(t, 20, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 2 {
			p.Waits = []DecisionWait{{Request: "long", UntilUS: v.NowUS + 5}}
			p.BatchTokenCap = 1
		}
		return p
	})
	a, b := longWaitRequest(), longWaitRequest()
	b.ID = "other"
	b.InputTokens = []TokenID{10, 11, 12, 13, 14, 15}
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if len(*records) != 4 || a.State != StateCompleted || b.State != StateCompleted {
		t.Fatal("expired cap stranded a resident", len(*records), a.State, b.State)
	}
	stale := (*records)[1]
	if stale.Feedback.Status != "superseded" || stale.Feedback.Reason != "prefill_wait_expired_batch_cap" || len(stale.Feedback.Grants) != 0 || stale.Service.ExtraUS != 20 || stale.Feedback.AtUS != 50 {
		t.Fatal("expired cap was applied or its service refunded", stale)
	}
	if a.FirstTokenTime != 110 || b.FirstTokenTime != 110 {
		t.Fatal("unexpected compute after fresh cap", a.FirstTokenTime, b.FirstTokenTime)
	}
}

func TestResidentPrefillWaitTimeoutClearsRefsAndTimer(t *testing.T) {
	s, _, _ := residentWaitFixture(t, 0, pauseLong)
	r := longWaitRequest()
	r.Deadline = 25
	s.InjectArrival(r)
	drainService(t, s)
	if r.State != StateTimedOut || r.TTFTSet || s.Clock != 25 || s.KVCache.UsedBlocks() != 0 || s.decision.timer != nil || len(s.decision.waits) != 0 {
		t.Fatal("resident timeout kept refs/timer or published output", r.State, s.Clock)
	}
}

func TestResidentPrefillWaitFollowsCapacityOrExplicitRequeue(t *testing.T) {
	s, _, records := residentWaitFixture(t, 0, func(v DecisionView) DecisionPlan {
		p := pauseLong(v)
		if v.Version == 2 {
			p.RevisitAtUS = atUS(20)
		}
		if v.Version == 3 {
			p.Preemptions = []string{"long"}
		}
		return p
	})
	if err := s.EnableDecisionPrefillPreemption(); err != nil {
		t.Fatal(err)
	}
	r := longWaitRequest()
	s.InjectArrival(r)
	drainService(t, s)
	if r.State != StateCompleted || s.Metrics.PreemptionCount != 1 || s.KVCache.UsedBlocks() != 0 {
		t.Fatal("paused request failed after requeue", r.State)
	}
	for _, d := range *records {
		if d.View.NowUS >= 10 && d.View.NowUS < 40 && len(d.Feedback.Grants) != 0 {
			t.Fatal("requeue bypassed request wait", d)
		}
	}
}

func TestResidentPrefillWaitRejectsInvalidPlanBeforeMutation(t *testing.T) {
	for _, mode := range []string{"unknown", "repeat", "decode", "output", "disabled", "preempt", "zero_cap"} {
		t.Run(mode, func(t *testing.T) {
			s, _, _ := residentWaitFixture(t, 0, pauseLong)
			s.EnableDecisionPrefillPreemption()
			r := longWaitRequest()
			r.State, r.ProgressIndex = StateRunning, 2
			s.KVCache.AllocateKVBlocks(r, 0, 2, nil)
			s.reqNumComputedTokens[r.ID] = 2
			s.RunningBatch.Requests = []*Request{r}
			if mode == "decode" {
				r.ProgressIndex = 6
				r.TTFTSet = true
			}
			if mode == "output" {
				s.decision.events.firstOutput[r.ID] = true
			}
			if mode == "disabled" {
				s.decision.prefillWaits = false
			}
			before := s.decisionView(10)
			s.decision.policy = &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
				p := DecisionPlan{Version: v.Version, Waits: []DecisionWait{{Request: "long", UntilUS: 40}}}
				switch mode {
				case "unknown":
					p.Waits[0].Request = "missing"
				case "repeat":
					p.Waits = append(p.Waits, p.Waits[0])
				case "preempt":
					p.Preemptions = []string{"long"}
				case "zero_cap":
					p.TokenCaps = []DecisionTokenCap{{Request: "long", Tokens: 0}}
				}
				return p
			}}
			func() {
				defer func() {
					if recover() == nil {
						t.Fatal("invalid resident wait accepted")
					}
				}()
				s.prepareDecision(10)
			}()
			if !reflect.DeepEqual(before.Running, s.decisionView(10).Running) || len(s.decision.waits) != 0 || s.KVCache.UsedBlocks() != 1 || s.decision.timer != nil {
				t.Fatal("rejected resident wait mutated resources")
			}
		})
	}
}

func TestResidentPrefillWaitCostsRequireExplicitCoverage(t *testing.T) {
	v := DecisionView{Capabilities: DecisionCapabilities{PrefillWaits: true}, KV: DecisionKVState{TransfersKnown: true}, Running: []DecisionRequest{{ID: "long", InputTokens: 6}}}
	p := DecisionPlan{Waits: []DecisionWait{{Request: "long", UntilUS: 40}}}
	m := LinearDecisionCost{PerPrefillWaitUS: 7, Provenance: "synthetic hold update", Limits: &DecisionCostLimits{MaxVisibleRequests: 2, MaxInputTokens: 6}}
	if _, err := m.Estimate(v, p); err == nil || !strings.Contains(err.Error(), "prefill waits") {
		t.Fatal("old profile silently covered new observation work", err)
	}
	m.Limits.RequirePrefillWaits = true
	fee, err := m.Estimate(v, p)
	if err != nil || fee.ExtraUS != 7 {
		t.Fatal("explicit update fee not counted", fee, err)
	}
}
