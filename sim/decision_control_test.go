package sim

import "testing"

type controlFixtureEvent struct {
	at  int64
	run func()
}

func (e *controlFixtureEvent) Timestamp() int64   { return e.at }
func (e *controlFixtureEvent) Priority() int      { return PriorityAdapterLoad }
func (e *controlFixtureEvent) Execute(*Simulator) { e.run() }

type controlFixtureStore struct {
	KVStore
	s              *Simulator
	remaining      int
	begins, adopts []int64
	physicalDone   bool
}

func (*controlFixtureStore) ExecutesEmptySteps() bool     { return true }
func (k *controlFixtureStore) HasPendingEngineWork() bool { return k.remaining > 0 }
func (k *controlFixtureStore) DecisionState([]DecisionKVQuery) DecisionKVState {
	v := DecisionKVState{PrefixStateKnown: true, TransfersKnown: true}
	if k.remaining > 0 {
		v.PendingTransfers = []DecisionTransfer{{ID: 1, Source: "hbm", Destination: "dram", Bytes: 16}}
	}
	return v
}
func (k *controlFixtureStore) BeginBatch(now int64, work []BatchWork, done func(int64)) (bool, error) {
	if len(work) > 0 || k.remaining == 0 {
		return false, nil
	}
	k.begins = append(k.begins, now)
	k.s.Schedule(&controlFixtureEvent{at: now + 3, run: func() {
		k.remaining--
		k.adopts = append(k.adopts, now+3)
		done(now + 3)
	}})
	return true, nil
}

func controlFixture(t *testing.T, cost int64, enabled bool, pending int) (*Simulator, *controlFixtureStore, *[]DecisionRecord) {
	t.Helper()
	s, records := serviceFixture(t, LinearDecisionCost{FixedUS: cost, Provenance: "control fixture"})
	s.RunningBatch = nil
	k := &controlFixtureStore{KVStore: s.KVCache, s: s, remaining: pending}
	s.KVCache = k
	if enabled {
		if err := s.EnableDecisionControlSteps(); err != nil {
			t.Fatal(err)
		}
	}
	return s, k, records
}

func TestControlDecisionPaysServiceBeforeIOWithoutInventingCleanup(t *testing.T) {
	s, k, records := controlFixture(t, 10, true, 1)
	s.Step(0)
	s.Schedule(&StepEvent{time: 3}) // redundant wake cannot duplicate service
	s.Schedule(&controlFixtureEvent{at: 5, run: func() { k.physicalDone = true }})
	for s.PeekNextEventTime() < 10 {
		s.ProcessNextEvent()
		if len(k.begins) != 0 || k.remaining != 1 {
			t.Fatal("I/O progressed before CPU service")
		}
	}
	drainService(t, s)
	if !k.physicalDone || len(k.begins) != 1 || k.begins[0] != 10 || k.adopts[0] != 13 || s.Clock != 13 {
		t.Fatalf("wrong control timeline: begin %v adopt %v end %d", k.begins, k.adopts, s.Clock)
	}
	if len(*records) != 1 {
		t.Fatalf("I/O completion alone must not invent another cleanup, got %d", len(*records))
	}
	for i, r := range *records {
		if !r.ControlStep || len(r.View.Waiting)+len(r.View.Running) != 0 || len(r.Feedback.Grants) != 0 || r.Service.ExtraUS != 10 || r.Feedback.Reason != "control_step" || r.View.Version != uint64(i+1) {
			t.Fatal("incorrect control evidence")
		}
	}
	if len(s.Metrics.RequestTTFTs) != 0 || s.RunningBatch != nil || s.stepEvent != nil || s.decision.pending != nil || s.pendingDecisionControl() {
		t.Fatal("control-only work left requests, metrics or an orphan continuation")
	}
}

func TestControlDecisionArrivalWaitsForFreshPlanAfterIO(t *testing.T) {
	s, k, records := controlFixture(t, 10, true, 1)
	r := serviceRequest("arrived", 5)
	s.InjectArrival(r)
	s.Step(0)
	drainService(t, s)
	if len(k.begins) != 1 || k.begins[0] != 10 || r.State != StateCompleted || s.Metrics.RequestTTFTs[r.ID] != 28 {
		t.Fatalf("late request joined control work: %v %s %v", k.begins, r.State, s.Metrics.RequestTTFTs)
	}
	if len(*records) != 3 || !(*records)[0].ControlStep || (*records)[1].ControlStep || (*records)[1].View.NowUS != 13 || len((*records)[1].Feedback.Grants) != 1 || !(*records)[2].ControlStep {
		t.Fatalf("late arrival did not get a distinct decision: %+v", *records)
	}
	if s.Clock != 43 || s.pendingDecisionControl() {
		t.Fatal("completion cleanup did not drain")
	}
}

func TestControlDecisionTimeoutCoalescesCleanupWithoutDroppingIO(t *testing.T) {
	s, k, records := controlFixture(t, 10, true, 1)
	r := serviceRequest("timeout", 5)
	r.Deadline = 8
	s.InjectArrival(r)
	s.Step(0)
	drainService(t, s)
	if r.State != StateTimedOut || len(k.begins) != 1 || k.adopts[0] != 13 || len(*records) != 2 || s.Clock != 23 {
		t.Fatalf("timeout abandoned I/O or repeated cleanup: %s %v %d %d", r.State, k.adopts, len(*records), s.Clock)
	}
	if _, ok := s.Metrics.RequestTTFTs[r.ID]; ok {
		t.Fatal("timed-out request gained a first token")
	}
}

func TestControlDecisionsZeroCostPreservesIOAndDefaultIsInert(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		s, k, records := controlFixture(t, 0, enabled, 1)
		s.Step(0)
		drainService(t, s)
		if len(k.begins) != 1 || k.begins[0] != 0 || k.adopts[0] != 3 || s.Clock != 3 {
			t.Fatal("zero fee changed actual I/O timing")
		}
		want := 0
		if enabled {
			want = 1
		}
		if len(*records) != want {
			t.Fatal("control policy called without opting in, or omitted after opting in")
		}
	}
}

func TestControlDecisionOnlyRunsForObservedWork(t *testing.T) {
	s, _, records := controlFixture(t, 10, true, 0)
	s.Step(0)
	if len(*records) != 0 || s.HasPendingEvents() {
		t.Fatal("idle engine invented a control iteration")
	}
	r := serviceRequest("r", 0)
	s.InjectArrival(r)
	drainService(t, s)
	if len(*records) != 2 || (*records)[0].ControlStep || !(*records)[1].ControlStep || s.Clock != 30 {
		t.Fatalf("request completion cleanup missing: %+v", *records)
	}
	if s.Metrics.RequestTTFTs[r.ID] != 20 {
		t.Fatal("terminal cleanup changed already published first token")
	}
}
