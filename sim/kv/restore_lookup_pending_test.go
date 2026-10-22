package kv

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Use real background STORE jobs, including their delayed scheduler publication.
// A second request has an ordinary APC hit for block 0; its full final prompt
// block is still being saved, just like the native duplicate-prefix counterexample.
func pendingStoreFixture(t *testing.T, blocks int64) (*peerHarness, *PeerCache, *sim.Request) {
	t.Helper()
	h, s, _ := restoreDecisionFixture(t)
	s.KVCacheState = NewKVCacheState(blocks, 2)
	if err := s.EnableDecodeCapacityReservation(); err != nil {
		t.Fatal(err)
	}
	producer := &sim.Request{ID: "producer", InputTokens: []sim.TokenID{70, 71, 72, 73}, MaxOutputLen: 1}
	if !s.AllocateKVBlocks(producer, 0, 4, nil) {
		t.Fatal("producer allocation")
	}
	producer.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{producer})
	s.ReleaseKVBlocks(producer)
	reader := &sim.Request{ID: "reader", InputTokens: producer.InputTokens, MaxOutputLen: 1}
	if len(s.GetRequestCachedPrefix(reader).Blocks) != 1 || h.f.Pending() != 1 {
		t.Fatal("fixture did not retain APC and one grouped unpublished store")
	}
	return h, s, reader
}

func TestRestoreLookupPendingDefersBeforeCapacityAndResumesAfterPublication(t *testing.T) {
	for _, blocks := range []int64{2, 8} {
		t.Run(fmt.Sprint(blocks), func(t *testing.T) {
			h, s, r := pendingStoreFixture(t, blocks)
			free := s.FreeBlockCnt
			prefix := s.GetRequestCachedPrefix(r)
			for retry := 0; retry < 3; retry++ {
				if s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) {
					t.Fatal("computed the suffix while its STORE lookup was pending")
				}
				failure := s.LastAllocationFailure()
				if failure.Kind != "wait" || failure.Reason != "restore_lookup_pending" || !s.IsDeferred(r.ID) {
					t.Fatalf("pending lookup became capacity pressure: %+v", failure)
				}
				if s.FreeBlockCnt != free || h.f.Pending() != 1 || len(s.decodeTargets) != 0 || len(s.holds) != 0 || s.readPending[r.ID] != 0 {
					t.Fatal("lookup wait allocated, reserved or submitted work")
				}
			}
			finishRestoreStep(t, h, s)
			if s.IsDeferred(r.ID) || h.f.Pending() != 0 {
				t.Fatal("publication did not wake the lookup")
			}
			prefix = s.GetRequestCachedPrefix(r)
			if s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) || s.LastAllocationFailure().Reason != "restore_submitted" {
				t.Fatal("published suffix did not enter the real LOAD path")
			}
			finishRestoreStep(t, h, s)
			prefix = s.GetRequestCachedPrefix(r)
			if prefix.Tokens != 3 || !s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) {
				t.Fatal("adopted suffix did not permit logits replay")
			}
			s.ReleaseKVBlocks(r)
			assertPeerConservation(t, s)
			if s.FreeBlockCnt != blocks || h.f.Pending() != 0 || len(s.decodeTargets) != 0 || len(s.holds) != 0 {
				t.Fatal("completed lookup/load leaked ownership")
			}
		})
	}
}

func TestRestoreLookupPendingRespectsExplicitRecomputeAndCancellation(t *testing.T) {
	h, s, r := pendingStoreFixture(t, 8)
	prefix := s.GetRequestCachedPrefix(r)
	if s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) || !s.IsDeferred(r.ID) {
		t.Fatal("fixture did not defer reader")
	}
	s.ClearDeferred(r.ID)
	if s.IsDeferred(r.ID) || h.f.Pending() != 1 {
		t.Fatal("cancellation cancelled another request's STORE")
	}
	// A newly applied restore endpoint before the pending suffix explicitly
	// chooses recomputation. The unrelated in-flight STORE must not hold it.
	s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: 1}})
	if !s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) {
		t.Fatal("pending block outside the selected restore range blocked compute")
	}
	s.ReleaseKVBlocks(r)
	finishRestoreStep(t, h, s)
	assertPeerConservation(t, s)
	if s.IsDeferred(r.ID) || len(s.decodeTargets) != 0 || len(s.holds) != 0 || s.FreeBlockCnt != 8 {
		t.Fatal("late STORE publication revived a cancelled lookup")
	}
}

func TestRestoreLookupPendingNeedsSchedulerPublication(t *testing.T) {
	h, s, r := pendingStoreFixture(t, 8)
	s.fabric.phases.model = fixedEnginePhases{EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 1, TailUS: 2}, 3}
	// The first worker poll precedes physical completion of the grouped store.
	finishRestoreStep(t, h, s)
	for h.f.active > 0 {
		phaseNext(t, h)
	}
	prefix := s.GetRequestCachedPrefix(r)
	if s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) || s.LastAllocationFailure().Reason != "restore_lookup_pending" {
		t.Fatal("physical completion was mistaken for scheduler publication")
	}
	finishRestoreStep(t, h, s)
	if s.IsDeferred(r.ID) || h.f.Pending() != 0 {
		t.Fatal("later scheduler adoption did not release lookup wait")
	}
	s.ClearDeferred(r.ID)
	assertPeerConservation(t, s)
}

func TestRestoreLookupFailureReportsReadyLoadWithoutReservingIt(t *testing.T) {
	h, s, r := restoreDecisionFixture(t)
	busy := &sim.Request{ID: "busy", InputTokens: []sim.TokenID{800, 801}}
	if !s.AllocateKVBlocks(busy, 0, 2, nil) {
		t.Fatal("fixture allocation")
	}
	if s.AllocateKVBlocks(r, 0, 2, nil) {
		t.Fatal("restore exceeded capacity")
	}
	f := s.LastAllocationFailure()
	if f.Kind != "capacity" || !f.RestoreLookupKnown || f.RestoreCandidateBlocks != 2 || h.f.Pending() != 0 || len(s.holds) != 0 {
		t.Fatal("failed LOAD lost its lookup or reserved work", f)
	}
	s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: 0}})
	if s.AllocateKVBlocks(r, 0, 2, nil) {
		t.Fatal("compute exceeded full-prefill capacity")
	}
	f = s.LastAllocationFailure()
	if !f.RestoreLookupKnown || f.RestoreCandidateBlocks != 0 {
		t.Fatal("previous LOAD lookup leaked into compute failure", f)
	}
	s.ReleaseKVBlocks(busy)
	assertPeerConservation(t, s)
}
