package sim

import "testing"

type testHostCosts struct{ registration int64 }

func (m testHostCosts) EstimateHostService(w HostServiceWork) (DecisionCostEstimate, error) {
	fee := m.registration
	if w.Stage == "completion" {
		fee = 10 + 5*int64(len(w.Requests)) + 4*int64(len(w.Finishing))
	}
	return DecisionCostEstimate{ExtraUS: fee, Provenance: "synthetic host components"}, nil
}

type hostReadyEvent struct {
	at  int64
	run func(int64)
}

func (e *hostReadyEvent) Timestamp() int64   { return e.at }
func (e *hostReadyEvent) Priority() int      { return PriorityStep }
func (e *hostReadyEvent) Execute(*Simulator) { e.run(e.at) }

type hostAsyncStore struct {
	KVStore
	s       *Simulator
	barrier HostCompletionBarrier
	starts  []int64
}

func (k *hostAsyncStore) BindHostCompletionBarrier(b HostCompletionBarrier) error {
	k.barrier = b
	return nil
}
func (k *hostAsyncStore) BeginBatch(now int64, w []BatchWork, done func(int64)) (bool, error) {
	k.starts = append(k.starts, now)
	k.s.Schedule(&hostReadyEvent{at: now + 10, run: func(at int64) { k.barrier(at, w, done) }})
	return true, nil
}

func hostFixture(t *testing.T, registration int64) (*Simulator, *hostAsyncStore, *[]HostServiceRecord) {
	s, _ := serviceFixture(t, LinearDecisionCost{FixedUS: 5, Provenance: "synthetic decision"})
	k := &hostAsyncStore{KVStore: s.KVCache, s: s}
	s.KVCache = k
	if err := s.SetExecutionCostModel(constantExecutionCost(7)); err != nil {
		t.Fatal(err)
	}
	var records []HostServiceRecord
	if err := s.SetHostServiceCosts(testHostCosts{registration}, true, true, func(r HostServiceRecord) { records = append(records, r) }); err != nil {
		t.Fatal(err)
	}
	return s, k, &records
}

func TestHostServicesSerializeRegistrationAndPublishAfterCompletion(t *testing.T) {
	s, k, records := hostFixture(t, 8)
	a, b := serviceRequest("a", 0), serviceRequest("b", 25)
	s.InjectArrival(a)
	s.InjectArrival(b)
	s.Schedule(&StepEvent{time: 35})
	for s.PeekNextEventTime() < 49 {
		s.ProcessNextEvent()
		if a.TTFTSet || a.ProgressIndex != 0 {
			t.Fatal("result visible before completion service ended")
		}
	}
	drainService(t, s)
	if a.FirstTokenTime != 49 || b.FirstTokenTime != 73 || len(k.starts) != 2 || k.starts[0] != 20 || k.starts[1] != 69 {
		t.Fatal("registration overlapped engine or completion published early", a.FirstTokenTime, b.FirstTokenTime, k.starts)
	}
	if len(*records) != 4 || (*records)[2].Work.Stage != "registration" || (*records)[2].Service.StartUS != 49 || (*records)[1].Work.Finishing[0] != "a" {
		t.Fatal("missing host-boundary evidence", *records)
	}
	if s.stepEvent != nil || s.hostServices.pending != nil || s.pendingRegistrations() || s.KVCache.UsedBlocks() != 0 {
		t.Fatal("host work or leases leaked")
	}
}

func TestHostCompletionCancellationKeepsCostAndSuppressesOutput(t *testing.T) {
	s, _, records := hostFixture(t, 8)
	a := serviceRequest("a", 0)
	a.Deadline = 40
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateTimedOut || a.TTFTSet || s.Clock != 49 || s.KVCache.UsedBlocks() != 0 || s.hostServices.pending != nil {
		t.Fatal("cancelled completion refunded, published or leaked", a.State, s.Clock)
	}
	if len(*records) != 2 || (*records)[1].Service.ExtraUS != 19 {
		t.Fatal("accepted completion work not paid")
	}
}

func TestRegistrationTimeoutCannotResurrectRequest(t *testing.T) {
	s, k, records := hostFixture(t, 8)
	a := serviceRequest("a", 0)
	a.Deadline = 5
	s.InjectArrival(a)
	drainService(t, s)
	if a.State != StateTimedOut || a.TTFTSet || s.Clock != 8 || len(k.starts) != 0 || len(*records) != 1 || s.WaitQ.Len() != 0 || s.stepEvent != nil {
		t.Fatal("registration resurrected cancellation or refunded CPU", a.State, s.Clock)
	}
}

func TestZeroRegistrationDuringCompletionPreservesImmediateAdmission(t *testing.T) {
	s, k, records := hostFixture(t, 0)
	a, b := serviceRequest("a", 0), serviceRequest("b", 30)
	s.InjectArrival(a)
	s.InjectArrival(b)
	drainService(t, s)
	if a.State != StateCompleted || b.State != StateCompleted || len(k.starts) != 2 || len(*records) != 4 {
		t.Fatal("zero registration overlapped or stranded a host operation")
	}
	if (*records)[2].Work.Stage != "registration" || (*records)[2].Service.StartUS != 30 || (*records)[2].Service.EndUS != 30 {
		t.Fatal("zero registration changed old admission boundary")
	}
}
