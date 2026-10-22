package sim

import "testing"

type testPostStepCosts struct{ output, idle int64 }

func (m testPostStepCosts) EstimatePostStep(w PostStepWork) (DecisionCostEstimate, error) {
	cost := m.output
	if w.Stage == "idle_check" {
		cost = m.idle
	}
	// An estimator must not be able to modify the observation sent to the log.
	if len(w.Requests) > 0 {
		w.Requests[0] = "mutated-estimate"
	}
	return DecisionCostEstimate{ExtraUS: cost, Provenance: "synthetic post-step boundary"}, nil
}

func postStepFixture(t *testing.T, n int, costs testPostStepCosts) (*Simulator, *[]PostStepRecord) {
	s, _ := serviceFixture(t, nil)
	var records []PostStepRecord
	if err := s.SetPostStepCostModel(costs, n, func(r PostStepRecord) { records = append(records, r) }); err != nil {
		t.Fatal(err)
	}
	return s, &records
}

func TestPostStepKeepsPublishedTTFTAndSerializesLateArrival(t *testing.T) {
	s, records := postStepFixture(t, 2, testPostStepCosts{20, 7})
	a, b := serviceRequest("a", 0), serviceRequest("b", 15)
	s.InjectArrival(a)
	s.InjectArrival(b)
	s.Schedule(&StepEvent{time: 20})
	drainService(t, s)
	if a.FirstTokenTime != 10 || b.FirstTokenTime+b.ArrivalTime != 47 || len(*records) != 3 {
		t.Fatal("post-output cost moved old TTFT or allowed overlapping execution", a.FirstTokenTime, b.FirstTokenTime, *records)
	}
	if (*records)[0].Service.StartUS != 10 || (*records)[1].Work.Stage != "idle_check" || len((*records)[1].Work.Requests) != 0 || (*records)[1].Service.EndUS != 37 {
		t.Fatal("driver input became registered before the next loop", *records)
	}
	if s.Clock != 67 || s.stepEvent != nil || s.postStep.active || s.postStep.pending != nil || s.postStep.remainingInputs != 0 {
		t.Fatal("post-step service failed to drain")
	}
}

func TestPostStepIdleCallAccountsForFutureInputWithoutChargingSleep(t *testing.T) {
	s, records := postStepFixture(t, 2, testPostStepCosts{20, 7})
	a, b := serviceRequest("a", 0), serviceRequest("b", 1000)
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if a.FirstTokenTime != 10 || b.FirstTokenTime != 10 || len(*records) != 3 || (*records)[1].Work.Stage != "idle_check" || len((*records)[1].Work.Requests) != 0 {
		t.Fatal("future input suppressed required idle check or sleep became CPU service", *records)
	}
}

func TestPostStepTimeoutDoesNotResurrectRemainingBatch(t *testing.T) {
	s, records := postStepFixture(t, 1, testPostStepCosts{30, 7})
	a := serviceRequest("a", 0)
	a.OutputTokens = []TokenID{9, 10}
	a.Deadline = 20
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateTimedOut || !a.TTFTSet || a.FirstTokenTime != 10 || s.Clock != 40 || s.KVCache.UsedBlocks() != 0 || s.RunningBatch != nil || s.stepEvent != nil || len(*records) != 1 {
		t.Fatal("post-output timeout restored a captured batch, refunded service, or moved TTFT", a.State, s.Clock, *records)
	}
}

func TestPostStepWaitExpiresDuringIdleServiceAndWakesOnce(t *testing.T) {
	s, records := postStepFixture(t, 1, testPostStepCosts{5, 20})
	p, _ := NewQueueDecisionPolicy("fcfs", 0)
	p.AdmissionDelayUS = 15
	if err := s.SetDecisionPolicy(p, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableDecisionPrefillWaits(); err != nil {
		t.Fatal(err)
	}
	a := serviceRequest("a", 0)
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateCompleted || a.FirstTokenTime != 35 || len(*records) != 3 || s.stepEvent != nil || s.postStep.active {
		t.Fatal("deadline was lost during post-step CPU service", a.State, a.FirstTokenTime, *records)
	}
	if len((*records)[1].Work.Held) != 1 || (*records)[1].Work.Requests[0] != "a" || (*records)[1].Service.StartUS != 5 || (*records)[1].Service.EndUS != 25 {
		t.Fatal("idle check did not use its actual starting wait state", *records)
	}
}

func TestPostStepDefersZeroHostRegistrationAndDrainsCancelledInput(t *testing.T) {
	for _, deadline := range []int64{0, 60} {
		s, k, host := hostFixture(t, 0)
		if err := s.SetPostStepCostModel(testPostStepCosts{20, 7}, 2, nil); err != nil {
			t.Fatal(err)
		}
		a, b := serviceRequest("a", 0), serviceRequest("b", 15)
		b.Deadline = deadline
		s.InjectArrival(a)
		s.InjectArrival(b)
		drainService(t, s)
		if a.FirstTokenTime != 41 || s.pendingRegistrations() || s.postStep.active || s.stepEvent != nil {
			t.Fatal("post-step registration leaked or moved published output", a.FirstTokenTime)
		}
		if deadline == 0 {
			if b.State != StateCompleted || b.FirstTokenTime+b.ArrivalTime != 109 || len(*host) != 4 || (*host)[2].Service.StartUS != 68 {
				t.Fatal("zero-cost host registration bypassed synchronous input handoff", *host)
			}
		} else if b.State != StateTimedOut || b.TTFTSet || len(*host) != 2 || len(k.starts) != 1 || s.KVCache.UsedBlocks() != 0 {
			t.Fatal("cancelled driver input was registered or executed", b.State, *host)
		}
	}
}
