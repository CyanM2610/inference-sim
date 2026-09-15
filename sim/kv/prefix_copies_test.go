package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func duplicatePrefills(t *testing.T, s *PeerCache) (*sim.Request, *sim.Request, int64, int64) {
	t.Helper()
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3}, MaxOutputLen: 1}
	b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{1, 2, 4}, MaxOutputLen: 1}
	// Two allocations before either computation completes create native-style
	// private duplicate pages, not a fabricated cache-index insertion.
	if !s.AllocateKVBlocks(a, 0, 2, nil) || !s.AllocateKVBlocks(b, 0, 2, nil) {
		t.Fatal("duplicate producers did not fit")
	}
	if len(s.GetCachedBlocks(a.InputTokens)) != 0 {
		t.Fatal("allocation published uncomputed prefix")
	}
	a.ProgressIndex, b.ProgressIndex = 2, 2
	s.MirrorToCPU([]*sim.Request{a, b})
	return a, b, s.RequestMap[a.ID][0], s.RequestMap[b.ID][0]
}

func TestFirstPublishedCopyControlsRealCapacityAdmission(t *testing.T) {
	_, s, _ := newPeerHarness(t, false)
	if err := s.EnableFirstPublishedCopy(); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableDecodeCapacityReservation(); err != nil {
		t.Fatal(err)
	}
	a, b, first, second := duplicatePrefills(t, s)
	s.ReleaseKVBlocks(a)
	s.MirrorToCPU([]*sim.Request{b})
	if ids := s.GetCachedBlocks(a.InputTokens); len(ids) != 1 || ids[0] != first || s.Blocks[first].RefCount != 0 || s.Blocks[second].RefCount != 1 {
		t.Fatal("expected first ref-free copy even with a later active duplicate", ids)
	}
	c := &sim.Request{ID: "c", InputTokens: []sim.TokenID{1, 2, 8, 9, 10, 11}, MaxOutputLen: 1}
	if s.AllocateKVBlocks(c, 2, 4, s.GetCachedBlocks(c.InputTokens)) || s.LastAllocationFailure().Reason != "decode_capacity_reservation" {
		t.Fatal("ref-free duplicate was incorrectly charged as an already-shared hit")
	}
	// Runtime eviction of the first copy exposes the still-protected duplicate.
	s.invalidate(s.Blocks[first])
	if ids := s.GetCachedBlocks(c.InputTokens); len(ids) != 1 || ids[0] != second || !s.AllocateKVBlocks(c, 2, 4, ids) {
		t.Fatal("remaining active duplicate did not permit admission", ids)
	}
	s.ReleaseKVBlocks(b)
	s.ReleaseKVBlocks(c)
	assertPeerConservation(t, s)
	if len(s.decodeTargets) != 0 {
		t.Fatal("capacity promises leaked")
	}
}

func TestFirstPublishedCopyEvictionFallbackAndSnapshot(t *testing.T) {
	_, s, _ := newPeerHarness(t, false)
	s.policy = BuiltinPeerPolicy{Name: "lru_drop"}
	if err := s.EnableFirstPublishedCopy(); err != nil {
		t.Fatal(err)
	}
	a, b, first, second := duplicatePrefills(t, s)
	// Re-publication in reverse order must not reorder the two copies.
	s.MirrorToCPU([]*sim.Request{b, a})
	if s.GetCachedBlocks(a.InputTokens)[0] != first {
		t.Fatal("publication changed insertion order")
	}
	s.ReleaseKVBlocks(a)
	frozen := s.SnapshotCachedBlocksFn()
	filler := &sim.Request{ID: "filler", InputTokens: []sim.TokenID{20, 21, 22, 23}}
	if !s.AllocateKVBlocks(filler, 0, 4, nil) {
		t.Fatal("filler did not fit")
	}
	last := &sim.Request{ID: "last", InputTokens: []sim.TokenID{30, 31}}
	if !s.AllocateKVBlocks(last, 0, 2, nil) {
		t.Fatal("oldest copy was not reclaimable")
	}
	if s.RequestMap[last.ID][0] != first || s.GetCachedBlocks(a.InputTokens)[0] != second {
		t.Fatal("real allocation lost the surviving duplicate")
	}
	s.ReleaseKVBlocks(b)
	s.invalidate(s.Blocks[second])
	if len(s.GetCachedBlocks(a.InputTokens)) != 0 || frozen(a.InputTokens) != 1 || len(s.prefixCopies) != 0 {
		t.Fatal("last-copy removal or detached snapshot is incorrect")
	}
	s.ReleaseKVBlocks(filler)
	s.ReleaseKVBlocks(last)
	assertPeerConservation(t, s)
}

func TestFirstPublishedComputeAndRestoreCopiesRemainSeparate(t *testing.T) {
	h, s, remote := newPeerHarness(t, false)
	if err := s.EnableFirstPublishedCopy(); err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "compute", InputTokens: []sim.TokenID{1, 2, 3}}
	if !s.AllocateKVBlocks(r, 0, 2, nil) {
		t.Fatal("producer allocation")
	}
	seed := seedPeer(t, remote, "remote", r.InputTokens)
	remoteID := remote.GetCachedBlocks(seed.InputTokens)[0]
	if !remote.store(remote.Blocks[remoteID], "dram", "store") {
		t.Fatal("source store")
	}
	h.next(t)
	key := remote.Blocks[remoteID].Hash
	// Reclaim STORE invalidates the source; use the retained input hash instead.
	if key == "" {
		key = s.fullPrefixHashes(r.InputTokens)[0]
	}
	e, access := s.find(key)
	if e == nil || !s.fetch(key, "reader", e, access, nil, false) {
		t.Fatal("restore submission")
	}
	r.ProgressIndex = 2
	s.MirrorToCPU([]*sim.Request{r})
	first := s.RequestMap[r.ID][0]
	if len(s.prefixCopies[key]) != 1 {
		t.Fatal("in-flight target became a ready copy")
	}
	h.next(t)
	if len(s.prefixCopies[key]) != 2 || s.GetCachedBlocks(r.InputTokens)[0] != first || len(s.holds["reader"]) != 1 {
		t.Fatal("restore replaced first computed copy or lost its own target")
	}
	restored := s.holds["reader"][0]
	s.ReleaseKVBlocks(r)
	s.invalidate(s.Blocks[first])
	if s.GetCachedBlocks(r.InputTokens)[0] != restored {
		t.Fatal("restore fallback missing")
	}
	s.ClearDeferred("reader")
	assertPeerConservation(t, s)
}

func TestLegacyCopySelectionAndConfigurationBoundary(t *testing.T) {
	_, s, _ := newPeerHarness(t, false)
	a, b, _, second := duplicatePrefills(t, s)
	if s.GetCachedBlocks(a.InputTokens)[0] != second {
		t.Fatal("default selection changed")
	}
	if err := s.EnableFirstPublishedCopy(); err == nil {
		t.Fatal("changed selection after cache use")
	}
	s.ReleaseKVBlocks(a)
	s.ReleaseKVBlocks(b)
}
