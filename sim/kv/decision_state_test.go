package kv

import (
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestDecisionSnapshotDoesNotTouchCacheOrRevealUnobservedCompletion(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2})
	r := &sim.Request{ID: "load", InputTokens: tokens}
	s.clock = 100
	s.AllocateKVBlocks(r, 0, 2, nil)
	queries := []sim.DecisionKVQuery{{ID: r.ID, Input: tokens}}
	var notified []sim.DecisionEvent
	s.BindDecisionEvents(func(e sim.DecisionEvent) {
		v := s.DecisionState(queries)
		if v.Requests[0].TransferPending || v.Requests[0].LocalPrefixTokens != 4 {
			t.Fatal("notification preceded KV publication")
		}
		for _, pending := range v.PendingTransfers {
			if pending.ID == e.Transfer.ID {
				t.Fatal("notification preceded pending removal")
			}
		}
		notified = append(notified, e)
	})
	s.EnableHashWorkAccounting()
	s.TakeHashWork()
	before := s.PeerSnapshot()
	records := len(h.records)
	v := s.DecisionState(queries)
	if !reflect.DeepEqual(before, s.PeerSnapshot()) || len(h.records) != records || s.TakeHashWork().Calls != 0 {
		t.Fatal("snapshot mutated cache/trace/hash accounting")
	}
	v.Pools[0].CapacityBlocks = -1
	v.Requests[0].ID = "fake"
	if s.DecisionState(queries).Requests[0].ID != r.ID {
		t.Fatal("snapshot aliases live state")
	}
	var end int64
	s.BeginBatch(100, []sim.BatchWork{{Request: &sim.Request{ID: "compute"}, NewTokens: 1}}, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	a := s.DecisionState(queries)
	for h.f.active > 0 {
		phaseNext(t, h)
	} // completion occurs after the worker poll
	b := s.DecisionState(queries)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("policy learned device completion before scheduler adoption")
	}
	if !b.Requests[0].TransferPending || len(b.PendingTransfers) != 1 || b.Requests[0].LocalPrefixTokens != 0 {
		t.Fatal("pending reservation became a usable hit")
	}
	if len(notified) != 0 {
		t.Fatal("physical completion notified policy before adoption")
	}
	end = 0
	s.BeginBatch(180, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	c := s.DecisionState(queries)
	if c.Requests[0].TransferPending || c.Requests[0].LocalPrefixTokens != 4 || len(c.PendingTransfers) != 0 {
		t.Fatal("adopted result absent from view")
	}
	if len(notified) != 1 || notified[0].Transfer.Bytes != 200 || notified[0].OccurredUS != end {
		t.Fatal("grouped copy did not produce one adopted notification")
	}
}

type decisionServiceLatency struct{}

func (decisionServiceLatency) StepTime([]*sim.Request) int64    { return 10 }
func (decisionServiceLatency) QueueingTime(*sim.Request) int64  { return 0 }
func (decisionServiceLatency) OutputTokenProcessingTime() int64 { return 0 }
func (decisionServiceLatency) PostDecodeFixedOverhead() int64   { return 0 }

func TestDecisionServiceAllowsDMACompletionButStillRequiresAdoption(t *testing.T) {
	h, cache, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2})
	r := &sim.Request{ID: "load", ArrivalTime: 100, InputTokens: tokens, OutputTokens: []sim.TokenID{9}, State: sim.StateQueued}
	cache.SetClock(100)
	cache.AllocateKVBlocks(r, 0, 2, nil)
	var end int64
	cache.BeginBatch(100, []sim.BatchWork{{Request: &sim.Request{ID: "compute"}, NewTokens: 1}}, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	if end != 132 || h.f.active != 1 {
		t.Fatal("fixture did not leave DMA in flight")
	}

	cfg := sim.SimConfig{Horizon: math.MaxInt64, KVCacheConfig: sim.NewKVCacheConfig(6, 2, 0, 0, 0, 0), BatchConfig: sim.NewBatchConfig(1, 16, 0), BatchCompletionEvents: true, Seed: 42}
	s, err := sim.NewSimulator(cfg, cache, decisionServiceLatency{})
	if err != nil {
		t.Fatal(err)
	}
	cache.BindEvents(s.Schedule, s.ScheduleStepIfIdle)
	for _, e := range h.events {
		s.Schedule(e)
	}
	h.events = nil
	p, _ := sim.NewQueueDecisionPolicy("fcfs", 0)
	var records []sim.DecisionRecord
	if err := s.SetDecisionPolicy(p, func(r sim.DecisionRecord) { records = append(records, r) }); err != nil {
		t.Fatal(err)
	}
	if err := s.SetDecisionCostModel(sim.LinearDecisionCost{FixedUS: 100, Provenance: "synthetic DMA overlap fixture"}); err != nil {
		t.Fatal(err)
	}
	s.InjectArrivalAt(r, 132)
	queries := []sim.DecisionKVQuery{{ID: r.ID, Input: tokens}}
	var completedDuringService bool
	for n := 0; s.HasPendingEvents(); n++ {
		if n > 1000 {
			t.Fatal("service/DMA fixture failed to drain")
		}
		s.ProcessNextEvent()
		if s.Clock > 132 && s.Clock < 232 && h.f.active == 0 {
			completedDuringService = true
			v := cache.DecisionState(queries)
			if !v.Requests[0].TransferPending || v.Requests[0].LocalPrefixTokens != 0 || r.ProgressIndex != 0 || r.TTFTSet {
				t.Fatal("physical completion was consumed during decision service")
			}
		}
	}
	if !completedDuringService || r.State != sim.StateCompleted || len(records) < 2 {
		t.Fatal("missing overlapped completion or actual request execution")
	}
	if len(records[0].Feedback.Grants) != 0 || records[0].Feedback.AtUS != 232 || records[0].Feedback.Unselected[0].Reason != "kv_pending" {
		t.Fatal("service end bypassed scheduler adoption")
	}
	var adopted int64
	for _, e := range h.records {
		if e.Name == "transfer_adopted" && e.Request == r.ID && e.Source != cache.id {
			adopted = e.Time
			break
		}
	}
	if adopted <= 232 || records[1].View.KV.Requests[0].TransferPending || records[1].Feedback.AtUS <= adopted {
		t.Fatalf("next decision did not use adopted KV: adopted=%d records=%+v", adopted, records)
	}
	assertPeerConservation(t, cache)
}
