package kv

import (
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
	end = 0
	s.BeginBatch(180, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	c := s.DecisionState(queries)
	if c.Requests[0].TransferPending || c.Requests[0].LocalPrefixTokens != 4 || len(c.PendingTransfers) != 0 {
		t.Fatal("adopted result absent from view")
	}
}
