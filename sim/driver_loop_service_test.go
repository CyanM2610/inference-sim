package sim

import "testing"

type testDriverLoopCosts struct {
	testPostStepCosts
	prefix, exit int64
}

func (m testDriverLoopCosts) DriverLoopCostsEnabled() bool { return true }
func (m testDriverLoopCosts) EstimatePostStep(w PostStepWork) (DecisionCostEstimate, error) {
	fee := m.prefix
	if w.Stage == "loop_exit" {
		fee = m.exit
	} else if w.Stage != "before_step" {
		return m.testPostStepCosts.EstimatePostStep(w)
	}
	return DecisionCostEstimate{ExtraUS: fee, Provenance: "synthetic driver loop"}, nil
}

func loopFixture(t *testing.T, inputs int, prefix int64) (*Simulator, *[]PostStepRecord) {
	s, _ := serviceFixture(t, nil)
	var records []PostStepRecord
	if err := s.SetPostStepCostModel(testDriverLoopCosts{prefix: prefix, exit: 2}, inputs, func(r PostStepRecord) { records = append(records, r) }); err != nil {
		t.Fatal(err)
	}
	return s, &records
}

func countLoopStages(records []PostStepRecord) map[string]int {
	counts := map[string]int{}
	for _, r := range records {
		counts[r.Work.Stage]++
	}
	return counts
}

func TestDriverLoopPrefixIncludesArrivalDuringService(t *testing.T) {
	s, records := loopFixture(t, 2, 10)
	a, b := serviceRequest("a", 0), serviceRequest("b", 3)
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	counts := countLoopStages(*records)
	if a.FirstTokenTime != 20 || b.FirstTokenTime+b.ArrivalTime != 20 || counts["before_step"] != 1 || counts["loop_exit"] != 1 || s.Clock != 22 || s.postStep.loopOpen || !s.postStep.exited {
		t.Fatal("driver service admitted early, missed late input or repeated the loop", a.FirstTokenTime, b.FirstTokenTime, *records)
	}
}

func TestDriverLoopChargesOnceAcrossSeveralHostRegistrations(t *testing.T) {
	s, _, host := hostFixture(t, 8)
	var records []PostStepRecord
	if err := s.SetPostStepCostModel(testDriverLoopCosts{prefix: 6, exit: 2}, 2, func(r PostStepRecord) { records = append(records, r) }); err != nil {
		t.Fatal(err)
	}
	a, b := serviceRequest("a", 0), serviceRequest("b", 0)
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if a.State != StateCompleted || b.State != StateCompleted || countLoopStages(records)["before_step"] != 1 || (*host)[0].Service.StartUS != 6 || (*host)[1].Service.StartUS != 14 {
		t.Fatal("registration recursion repeated driver service", records, *host)
	}
}

func TestDriverLoopEmptyInputGapsAndFinalCondition(t *testing.T) {
	for _, initial := range []bool{false, true} {
		inputs, expectedLoops := 1, 2
		if initial {
			inputs, expectedLoops = 2, 3
		}
		s, records := loopFixture(t, inputs, 6)
		if initial {
			s.InjectArrival(serviceRequest("a", 0))
		}
		b := serviceRequest("b", 1000)
		s.InjectArrival(b)
		// An unrelated wake in the empty-input sleep must not rebill its loop.
		s.Schedule(&StepEvent{time: 500})
		drainService(t, s)
		counts := countLoopStages(*records)
		if b.FirstTokenTime != 16 || counts["before_step"] != expectedLoops || counts["after_output"] != inputs || counts["loop_exit"] != 1 || s.Clock != 1018 || s.stepEvent != nil {
			t.Fatal("empty loop, sleep or final condition miscounted", initial, b.FirstTokenTime, counts)
		}
	}
}

func TestDriverLoopTimeoutPaysPrefixWithoutInventingOutputWork(t *testing.T) {
	s, records := loopFixture(t, 1, 10)
	a := serviceRequest("a", 0)
	a.Deadline = 5
	s.InjectArrival(a)
	drainService(t, s)
	counts := countLoopStages(*records)
	if a.State != StateTimedOut || a.TTFTSet || counts["before_step"] != 1 || counts["loop_exit"] != 1 || counts["after_output"] != 0 || counts["idle_check"] != 0 || s.Clock != 12 || s.stepEvent != nil || s.pendingRegistrations() {
		t.Fatal("cancelled input created output work or refunded driver service", a.State, counts, s.Clock)
	}
}
