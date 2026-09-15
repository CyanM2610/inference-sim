package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

func TestGroupedCopiesPaySetupOnceAndPublishTogether(t *testing.T) {
	h, a, b := newPeerHarness(t, false)
	if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "cxl", GroupTransfers: true, RestoreWindow: 4}); err != nil {
		t.Fatal(err)
	}
	a.policy = BuiltinPeerPolicy{Name: "lru_drop"}
	b.policy = a.policy
	r := seedPeer(t, a, "writer", []sim.TokenID{1, 2, 3, 4, 5})
	if h.f.Pending() != 1 || h.f.Snapshot()["cxl"]["reserved"] != 2 {
		t.Fatal("two physical reservations must share one copy job")
	}
	if at := h.next(t); at != 43 {
		t.Fatalf("200 bytes / 5 + one 3us setup: got %d", at)
	}
	if h.f.Snapshot()["cxl"]["ready"] != 2 || len(a.GetCachedBlocks(r.InputTokens)) != 2 {
		t.Fatal("group did not retain/publish both copies")
	}
	b.SetClock(43)
	q := &sim.Request{ID: "reader", InputTokens: r.InputTokens}
	if b.AllocateKVBlocks(q, 0, 5, nil) || h.f.Pending() != 1 || len(b.GetCachedBlocks(q.InputTokens)) != 0 {
		t.Fatal("restore group published before completion")
	}
	if at := h.next(t); at != 86 {
		t.Fatalf("one grouped restore, got %d", at)
	}
	cached := b.GetCachedBlocks(q.InputTokens)
	if len(cached) != 2 || !b.AllocateKVBlocks(q, 4, 5, cached) {
		t.Fatal("restored prefix not consumable")
	}
	b.ReleaseKVBlocks(q)
	assertPeerConservation(t, a)
	assertPeerConservation(t, b)
	starts := 0
	for _, r := range h.records {
		if r.Name == "transfer_start" {
			starts++
			if r.Bytes != 200 || len(r.Hashes) != 2 {
				t.Fatal("group lost physical byte/hash provenance")
			}
		}
	}
	if starts != 2 {
		t.Fatalf("expected store and restore jobs, got %d", starts)
	}
}
