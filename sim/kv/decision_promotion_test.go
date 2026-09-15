package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestDecisionPromotionReservesDeduplicatesAndWaitsForAdoption(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 10, TailUS: 2})
	if err := s.EnablePromotionDecisions(); err != nil {
		t.Fatal(err)
	}
	q := sim.DecisionKVQuery{ID: "waiting", Input: tokens}
	first := s.PromoteRequestPrefix(q, 2)
	if first.StartedBlocks != 2 || first.BlockedBlocks != 0 || h.f.Pending() != 1 || s.FreeBlockCnt != 4 || len(s.GetCachedBlocks(tokens)) != 0 {
		t.Fatalf("promotion did not reserve one grouped unpublished copy: %+v pending=%d", first, h.f.Pending())
	}
	for _, id := range []string{"waiting", "shared"} {
		again := s.PromoteRequestPrefix(sim.DecisionKVQuery{ID: id, Input: tokens}, 2)
		if again.PendingBlocks != 2 || again.StartedBlocks != 0 || h.f.Pending() != 1 {
			t.Fatal("repeated/shared promotion submitted duplicate I/O", again)
		}
	}
	var end int64
	s.BeginBatch(100, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	for h.f.active > 0 {
		phaseNext(t, h)
	}
	if len(s.GetCachedBlocks(tokens)) != 0 || s.FreeBlockCnt != 4 {
		t.Fatal("device completion prematurely published or released promotion")
	}
	// Request cancellation must leave submitted source/target leases until
	// adoption. The copy may then populate the global cache for another reader.
	s.ClearDeferred(q.ID)
	end = 0
	s.BeginBatch(180, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	ready := s.PromoteRequestPrefix(sim.DecisionKVQuery{ID: "shared", Input: tokens}, 2)
	if ready.ReadyBlocks != 2 || ready.StartedBlocks != 0 || h.f.Pending() != 0 || s.FreeBlockCnt != 6 || len(s.holds[q.ID]) != 0 {
		t.Fatal("promotion adoption did not publish and release ownership", ready)
	}
	for _, p := range h.f.Snapshot() {
		if p["reserved"] != 0 || p["read_pins"] != 0 {
			t.Fatal("cancelled consumer leaked source pins", p)
		}
	}
	assertPeerConservation(t, s)
}

func TestDecisionPromotionReportsPartialCapacityAndMissingSource(t *testing.T) {
	for _, mode := range []string{"capacity", "missing_source"} {
		t.Run(mode, func(t *testing.T) {
			h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 10, TailUS: 2})
			if err := s.EnablePromotionDecisions(); err != nil {
				t.Fatal(err)
			}
			if mode == "capacity" {
				s.KVCacheState = NewKVCacheState(1, 2)
			} else {
				delete(h.f.pools["cxl"].entries, s.hashes(tokens)[1])
			}
			result := s.PromoteRequestPrefix(sim.DecisionKVQuery{ID: "waiting", Input: tokens}, 2)
			want := "hbm_capacity"
			if mode == "missing_source" {
				want = "source_not_ready"
			}
			if result.StartedBlocks != 1 || result.BlockedBlocks != 1 || result.Reason != want || h.f.Pending() != 1 {
				t.Fatal("partial action lost its submitted/blocked distinction", result)
			}
			var end int64
			s.BeginBatch(0, nil, func(at int64) { end = at })
			for end == 0 || h.f.active > 0 {
				phaseNext(t, h)
			}
			end = 0
			s.BeginBatch(100, nil, func(at int64) { end = at })
			for end == 0 {
				phaseNext(t, h)
			}
			if len(s.GetCachedBlocks(tokens)) != 1 || h.f.Pending() != 0 {
				t.Fatal("runtime implicitly retried blocked suffix")
			}
			assertPeerConservation(t, s)
		})
	}
}
