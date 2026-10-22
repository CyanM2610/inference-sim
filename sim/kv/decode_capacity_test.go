package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestDecodeCapacityReservesClientLimitWithoutFutureOutputs(t *testing.T) {
	for _, actual := range []int{1, 4} {
		_, s, _ := newPeerHarness(t, false)
		if err := s.EnableDecodeCapacityReservation(); err != nil {
			t.Fatal(err)
		}
		a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4}, OutputTokens: make([]sim.TokenID, actual), MaxOutputLen: 4}
		if !s.AllocateKVBlocks(a, 0, 2, nil) {
			t.Fatal("initial budget should fit")
		}
		a.ProgressIndex = 2
		if s.decodeTargets[a.ID] != 4 || s.PeerSnapshot()["decode_reserved_blocks"] != 3 || s.UsedBlocks() != 1 {
			t.Fatal("reservation read actual future output or allocated physical blocks")
		}
		b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{10, 11}, MaxOutputLen: 1}
		if s.AllocateKVBlocks(b, 0, 2, nil) || s.LastAllocationFailure().Reason != "decode_capacity_reservation" {
			t.Fatal("new admission consumed protected decode capacity")
		}
		if _, exists := s.decodeTargets[b.ID]; exists {
			t.Fatal("rejected request left a reservation")
		}
		if !s.AllocateKVBlocks(a, 2, 4, nil) {
			t.Fatal("reserved prefill cannot progress")
		}
		s.ReleaseKVBlocks(a)
		if len(s.decodeTargets) != 0 || s.PeerSnapshot()["decode_reserved_blocks"] != 0 {
			t.Fatal("completion/preemption leaked logical capacity")
		}
	}
}

func TestDecodeCapacityCountsSharedPrefixOnlyOnce(t *testing.T) {
	_, s, _ := newPeerHarness(t, false)
	if err := s.EnableDecodeCapacityReservation(); err != nil {
		t.Fatal(err)
	}
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, MaxOutputLen: 1}
	if !s.AllocateKVBlocks(a, 0, 4, nil) {
		t.Fatal("first shared producer failed")
	}
	a.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{a})
	b := &sim.Request{ID: "b", InputTokens: append([]sim.TokenID(nil), a.InputTokens...), MaxOutputLen: 1}
	cached := s.GetCachedBlocks(b.InputTokens)
	if len(cached) != 2 || !s.AllocateKVBlocks(b, 4, 5, cached) {
		t.Fatal("shared prefix incorrectly charged twice")
	}
	if s.UsedBlocks() != 3 || s.decodeCapacityReserved("") != 1 {
		t.Fatal("wrong physical/shared capacity reservation")
	}
	s.ReleaseKVBlocks(a)
	s.ReleaseKVBlocks(b)
	assertPeerConservation(t, s)
}

func TestDecodeCapacityCancellationClearsPendingLogicalPromise(t *testing.T) {
	h, s, _ := newPeerHarness(t, false)
	seedReclaimCache(t, s)
	if err := s.EnableDecodeCapacityReservation(); err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "waiting", InputTokens: []sim.TokenID{20, 21}, MaxOutputLen: 1}
	if s.AllocateKVBlocks(r, 0, 2, nil) || s.decodeTargets[r.ID] != 1 {
		t.Fatal("expected a pending writeback admission with logical capacity")
	}
	s.ClearDeferred(r.ID)
	if len(s.decodeTargets) != 0 {
		t.Fatal("cancelled waiting request kept its logical promise")
	}
	for len(h.events) > 0 {
		h.next(t)
	}
	if len(s.decodeTargets) != 0 || s.PeerSnapshot()["active_or_pinned"] != 0 {
		t.Fatal("writeback completion recreated a cancelled reservation")
	}
}
