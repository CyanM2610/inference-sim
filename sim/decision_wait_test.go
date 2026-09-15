package sim

import "testing"

type timerTestPolicy struct {
	decide func(DecisionView) DecisionPlan
	events []DecisionEvent
}

func (p *timerTestPolicy) Decide(v DecisionView) DecisionPlan { return p.decide(v) }
func (*timerTestPolicy) Observe(DecisionFeedback)             {}
func (p *timerTestPolicy) OnEvent(e DecisionEvent)            { p.events = append(p.events, e) }
func atUS(at int64) *int64                                    { return &at }

func waitFixture(t *testing.T, cost int64, decide func(DecisionView) DecisionPlan) (*Simulator, *timerTestPolicy, *[]DecisionRecord) {
	t.Helper()
	s, _ := serviceFixture(t, nil)
	p := &timerTestPolicy{decide: decide}
	var records []DecisionRecord
	if err := s.SetDecisionPolicy(p, func(r DecisionRecord) { records = append(records, r) }); err != nil {
		t.Fatal(err)
	}
	if cost > 0 {
		if err := s.SetDecisionCostModel(LinearDecisionCost{FixedUS: cost, Provenance: "synthetic timer test"}); err != nil {
			t.Fatal(err)
		}
	}
	return s, p, &records
}
func timerEvents(p *timerTestPolicy) []DecisionEvent {
	var events []DecisionEvent
	for _, e := range p.events {
		if e.Kind == "timer" {
			events = append(events, e)
		}
	}
	return events
}

func TestDecisionWaitAllRequestsWakeAndCoalesce(t *testing.T) {
	s, p, records := waitFixture(t, 0, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 1 {
			p.Waits = []DecisionWait{{"a", 40}, {"b", 40}}
			p.RevisitAtUS = atUS(20)
		}
		return p
	})
	a, b := serviceRequest("a", 0), serviceRequest("b", 0)
	s.InjectArrival(a)
	s.InjectArrival(b)
	timers := 0
	for n := 0; s.HasPendingEvents(); n++ {
		if n > 100 {
			t.Fatal("busy wait")
		}
		e := s.ProcessNextEvent()
		if _, ok := e.(*DecisionTimerEvent); ok {
			timers++
		}
		if s.Clock < 40 && (s.KVCache.UsedBlocks() != 0 || a.ProgressIndex != 0 || b.ProgressIndex != 0) {
			t.Fatal("waiting request allocated/computed before deadline")
		}
	}
	if timers != 2 || len(*records) != 3 || a.State != StateCompleted || b.State != StateCompleted || s.Metrics.RequestTTFTs["a"] != 50 {
		t.Fatalf("bad wait/revisit progression: timers=%d decisions=%d", timers, len(*records))
	}
	if len(timerEvents(p)) != 3 || (*records)[0].Feedback.Unselected[0].Reason != "policy_wait" || (*records)[1].View.Waiting[0].WaitUntilUS != 40 {
		t.Fatal("wait state/feedback missing")
	}
}

func TestDecisionWaitDoesNotBlockOtherRequestsOrInterruptBatch(t *testing.T) {
	s, p, records := waitFixture(t, 0, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 1 {
			p.Waits = []DecisionWait{{"a", 5}}
			p.RevisitAtUS = atUS(5)
		}
		return p
	})
	a, b := serviceRequest("a", 0), serviceRequest("b", 0)
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if s.Metrics.RequestTTFTs["b"] != 10 || s.Metrics.RequestTTFTs["a"] != 20 || len(*records) != 2 || (*records)[1].View.NowUS != 10 {
		t.Fatal("timer interrupted batch or held the eligible request")
	}
	if len(timerEvents(p)) != 2 || timerEvents(p)[0].OccurredUS != 5 {
		t.Fatal("missing in-flight timer")
	}
}

func TestDecisionWaitExpiryDuringServiceIsBuffered(t *testing.T) {
	s, p, records := waitFixture(t, 10, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 1 {
			p.Waits = []DecisionWait{{"a", 20}}
		}
		return p
	})
	a, b := serviceRequest("a", 0), serviceRequest("b", 15)
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	events := timerEvents(p)
	if len(events) != 1 || events[0].OccurredUS != 20 || events[0].DeliveredUS != 25 || len(*records) != 2 || (*records)[1].Feedback.AtUS != 25 || s.Metrics.RequestTTFTs["a"] != 35 {
		t.Fatal("expiry changed the active service or was delivered early")
	}
}

func TestDecisionWaitDeadlineElapsedBeforeApplicationDoesNotCreateTimer(t *testing.T) {
	s, p, records := waitFixture(t, 10, func(v DecisionView) DecisionPlan {
		return DecisionPlan{Version: v.Version, Waits: []DecisionWait{{"a", 5}}, RevisitAtUS: atUS(5)}
	})
	s.InjectArrival(serviceRequest("a", 0))
	drainService(t, s)
	if len(*records) != 1 || (*records)[0].Feedback.WaitUpdates[0].Status != "elapsed" || (*records)[0].Feedback.Revisit.Status != "elapsed" || len(timerEvents(p)) != 0 || s.Metrics.RequestTTFTs["a"] != 20 {
		t.Fatal("elapsed deadline was retroactively scheduled")
	}
}

func TestDecisionWaitReplaceAndClearRemoveOldTimers(t *testing.T) {
	s, p, records := waitFixture(t, 0, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		switch v.Version {
		case 1:
			p.Waits = []DecisionWait{{"a", 100}}
			p.RevisitAtUS = atUS(5)
		case 2:
			p.Waits = []DecisionWait{{"a", 50}}
			p.RevisitAtUS = atUS(10)
		case 3:
			p.Waits = []DecisionWait{{"a", 0}}
			p.RevisitAtUS = atUS(0)
		}
		return p
	})
	s.InjectArrival(serviceRequest("a", 0))
	drainService(t, s)
	if s.Clock != 20 || s.decision.timer != nil || len(s.decision.waits) != 0 || len(timerEvents(p)) != 2 || len(*records) != 3 || (*records)[2].Feedback.WaitUpdates[0].Status != "cleared" {
		t.Fatal("replaced/cancelled timer survived or changed final clock")
	}
}

func TestDecisionWaitTimeoutRemovesFarFutureTimers(t *testing.T) {
	s, p, _ := waitFixture(t, 0, func(v DecisionView) DecisionPlan {
		return DecisionPlan{Version: v.Version, Waits: []DecisionWait{{"a", 1000000}}, RevisitAtUS: atUS(2000000)}
	})
	a := serviceRequest("a", 0)
	a.Deadline = 5
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateTimedOut || s.Clock != 5 || s.HasPendingEvents() || s.decision.timer != nil || len(timerEvents(p)) != 0 {
		t.Fatal("orphaned timer inflated end time")
	}
}

func TestDecisionWaitSameTickTimeoutDiscardsInFlightResults(t *testing.T) {
	s, _, _ := waitFixture(t, 0, func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if v.Version == 1 {
			p.Waits = []DecisionWait{{"a", 5}}
		}
		return p
	})
	a := serviceRequest("a", 0)
	a.Deadline = 5
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateTimedOut || a.ProgressIndex != 0 || a.TTFTSet || s.Clock != 15 {
		t.Fatal("co-timed timeout resurrected output or discarded submitted compute duration")
	}
}

func TestDecisionWaitRejectsInvalidActionsBeforeMutation(t *testing.T) {
	for _, p := range []DecisionPlan{{Waits: []DecisionWait{{"future", 10}}}, {Waits: []DecisionWait{{"a", -1}}}, {Waits: []DecisionWait{{"a", 5}}}, {Waits: []DecisionWait{{"a", 10}, {"a", 11}}}, {RevisitAtUS: atUS(-1)}, {RevisitAtUS: atUS(5)}} {
		s, _, records := waitFixture(t, 0, func(v DecisionView) DecisionPlan { p.Version = v.Version; return p })
		s.WaitQ.Enqueue(serviceRequest("a", 0))
		func() {
			defer func() {
				if recover() == nil {
					t.Error("invalid deadline accepted")
				}
			}()
			s.Step(5)
		}()
		if s.KVCache.UsedBlocks() != 0 || s.decision.timer != nil || len(s.decision.waits) != 0 || len(*records) != 1 || (*records)[0].Feedback.Status != "rejected" {
			t.Fatal("rejected wait caused partial mutation")
		}
	}
}

func TestDecisionWaitCannotPauseRunningOrUnsupportedBackend(t *testing.T) {
	v := DecisionView{Version: 1, NowUS: 5, Running: []DecisionRequest{{ID: "running"}}, Capabilities: DecisionCapabilities{WaitUntil: true}}
	if err := ValidateDecision(v, DecisionPlan{Version: 1, Waits: []DecisionWait{{"running", 10}}}); err == nil {
		t.Fatal("running request can be paused without preemption")
	}
	v.Waiting = v.Running
	v.Running = nil
	v.Capabilities.WaitUntil = false
	if err := ValidateDecision(v, DecisionPlan{Version: 1, Waits: []DecisionWait{{"running", 10}}}); err == nil {
		t.Fatal("legacy backend silently accepts wait")
	}
}
