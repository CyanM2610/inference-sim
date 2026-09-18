package kv

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Observed on installed vLLM 5e7136e30027 with its real scheduler, allocator
// and CPU manager. Evidence: experiment/[2026.09.18]restore-liveness/native-cpu-v2.
// The second case exceeds both the old 32-block window and 48-block profile.
func TestNativeRestoreAdmissionMatchesServerAllocation(t *testing.T) {
	for _, c := range []struct{ capacity, a, b, loaded, free, reserved int64 }{
		{7, 95, 63, 5, 2, 1}, {80, 1119, 719, 69, 11, 1},
	} {
		t.Run(fmt.Sprint(c.capacity), func(t *testing.T) {
			h, s := reclaimReuseFixture(t, c.capacity)
			s.KVCacheState = NewKVCacheState(c.capacity, 16)
			h.f.pools["cpu"].config.CapacityBlocks = 256
			h.f.mechanisms.RestoreWindow = 32
			if err := s.EnableRestoreDecisions(); err != nil {
				t.Fatal(err)
			}
			if err := s.EnableNativeRestoreAdmission(); err != nil {
				t.Fatal(err)
			}
			requests := []*sim.Request{{ID: "a", InputTokens: make([]sim.TokenID, c.a)}, {ID: "b", InputTokens: make([]sim.TokenID, c.b)}}
			for i, r := range requests {
				for j := range r.InputTokens {
					r.InputTokens[j] = sim.TokenID(100 + i)
				}
				for j, key := range s.fullPrefixHashes(r.InputTokens) {
					e, fresh := h.f.reserve("cpu", key, r.InputTokens[16*j:16*j+16], 0)
					if !fresh {
						t.Fatal("CPU fixture reservation failed")
					}
					e.ready = true
				}
			}
			for _, r := range requests {
				if s.AllocateKVBlocks(r, 0, 32, nil) {
					t.Fatal("unexpected compute admission")
				}
			}
			if int64(s.readPending["a"]) != c.loaded || s.FreeBlockCnt != c.free || s.otherPrefillReservations("b") != c.reserved {
				t.Fatal("simulator disagrees with installed vLLM first admission", s.readPending, s.FreeBlockCnt, s.prefillTargets)
			}
			if s.readPending["b"] != 0 || len(s.holds["b"]) != 0 || s.prefillTargets["b"] != 0 || s.IsDeferred("b") {
				t.Fatal("native-rejected follower became a partial/deferred owner")
			}
		})
	}
}

func TestNativeRestoreAdmissionPreventsWindowedHoldAndWait(t *testing.T) {
	h, s := reclaimReuseFixture(t, 7)
	h.f.mechanisms.RestoreWindow = 1
	if err := s.EnableRestoreDecisions(); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableNativeRestoreAdmission(); err != nil {
		t.Fatal(err)
	}
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}
	b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{21, 22, 23, 24, 25, 26, 27}}
	for _, r := range []*sim.Request{a, b} {
		for i, hash := range s.fullPrefixHashes(r.InputTokens) {
			e, fresh := h.f.reserve("cpu", hash, r.InputTokens[2*i:2*i+2], 0)
			if !fresh {
				t.Fatal("fixture CPU reservation")
			}
			e.ready = true
		}
	}
	if s.AllocateKVBlocks(a, 0, 2, nil) || s.readPending[a.ID] != 5 {
		t.Fatalf("native lookup was split into partial owners: pending=%d, want=5", s.readPending[a.ID])
	}
	if s.AllocateKVBlocks(b, 0, 2, nil) || s.readPending[b.ID] != 0 || len(s.holds[b.ID]) != 0 || s.prefillTargets[b.ID] != 0 {
		t.Fatal("rejected follower retained a partial reservation")
	}
	if s.FreeBlockCnt != 2 || s.otherPrefillReservations(b.ID) != 1 {
		t.Fatal("wrong physical allocation or remaining first-request reservation")
	}
	// Drive only real engine events and pending transfer work. No new arrivals
	// or perpetual retry timer can rescue this fixture.
	now := int64(0)
	for _, r := range []*sim.Request{a, b} {
		admitted := false
		for tries := 0; tries < 128 && !admitted; tries++ {
			prefix := s.GetRequestCachedPrefix(r)
			admitted = s.AllocateKVBlocks(r, prefix.Tokens, r.InputLen(), prefix.Blocks)
			if admitted {
				break
			}
			if !s.HasPendingEngineWork() {
				t.Fatal("event queue exhausted with a retained restore owner", r.ID, s.LastAllocationFailure())
			}
			end := int64(-1)
			if ok, err := s.BeginBatch(now, nil, func(at int64) { end = at }); !ok || err != nil {
				t.Fatal(ok, err)
			}
			for end < 0 {
				phaseNext(t, h)
			}
			now = end
			s.SetClock(now)
		}
		if !admitted {
			t.Fatal("restore never reached compute admission", r.ID)
		}
		s.ReleaseKVBlocks(r)
	}
	if s.FreeBlockCnt != 7 || len(s.holds) != 0 || len(s.prefillTargets) != 0 || h.f.Pending() != 0 {
		t.Fatal("restore admission leaked capacity")
	}
	assertPeerConservation(t, s)
}
