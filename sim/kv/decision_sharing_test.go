package kv

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestSharingSnapshotPreservesPrefixIdentityAndAdoptionBoundary(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2})
	other := append([]sim.TokenID(nil), tokens...)
	other[0]++ // Equal later token blocks with a different prefix are different KV.
	q := []sim.DecisionKVQuery{{ID: "owner", Input: tokens}, {ID: "follower", Input: tokens}, {ID: "other", Input: other}}
	off := s.DecisionState(q)
	if off.PrefixSharingKnown || len(off.Requests[0].PrefixBlocks) != 0 {
		t.Fatal("default snapshot exposed extra work")
	}
	before := s.PeerSnapshot()
	on := s.DecisionStateWithSharing(q)
	if !on.PrefixSharingKnown || !reflect.DeepEqual(before, s.PeerSnapshot()) {
		t.Fatal("sharing snapshot mutated runtime")
	}
	for i, block := range on.Requests[0].PrefixBlocks {
		if block.ID != on.Requests[1].PrefixBlocks[i].ID || block.ID == on.Requests[2].PrefixBlocks[i].ID || block.LoadPending {
			t.Fatal("sharing lost chained content identity", on)
		}
	}
	on.Requests[0].PrefixBlocks[0].ID = "fake"
	if s.DecisionStateWithSharing(q).Requests[0].PrefixBlocks[0].ID == "fake" {
		t.Fatal("snapshot mutated backend identity")
	}
	s.clock = 100
	s.AllocateKVBlocks(&sim.Request{ID: "owner", InputTokens: tokens}, 0, 2, nil)
	pending := s.DecisionStateWithSharing(q)
	if !pending.Requests[1].PrefixBlocks[0].LoadPending || pending.Requests[1].TransferPending || pending.Requests[1].LocalPrefixBlocks != 0 {
		t.Fatal("shared follower did not see dependency", pending)
	}
	var end int64
	s.BeginBatch(100, []sim.BatchWork{{Request: &sim.Request{ID: "compute"}, NewTokens: 1}}, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	pre := s.DecisionStateWithSharing(q)
	for h.f.active > 0 {
		phaseNext(t, h)
	}
	if !reflect.DeepEqual(pre, s.DecisionStateWithSharing(q)) {
		t.Fatal("physical completion leaked into sharing state")
	}
	end = 0
	s.BeginBatch(180, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	adopted := s.DecisionStateWithSharing(q)
	if adopted.Requests[1].PrefixBlocks[0].LoadPending || adopted.Requests[1].LocalPrefixBlocks != 2 {
		t.Fatal("adoption did not publish shared prefix", adopted)
	}
	assertPeerConservation(t, s)
}
