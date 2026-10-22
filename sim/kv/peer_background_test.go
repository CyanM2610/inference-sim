package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestPeerBackgroundStoreRetainsHBMAndPublishesOnlyOnCompletion(t *testing.T) {
	h, a, _ := newPeerHarness(t, false)
	if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "cxl"}); err != nil {
		t.Fatal(err)
	}
	a.policy = BuiltinPeerPolicy{Name: "lru_drop"}
	r := &sim.Request{ID: "writer", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}}
	if !a.AllocateKVBlocks(r, 0, 5, nil) {
		t.Fatal("allocate")
	}
	a.MirrorToCPU([]*sim.Request{r})
	if h.f.Pending() != 0 {
		t.Fatal("uncomputed blocks were copied")
	}
	r.ProgressIndex = 5
	a.MirrorToCPU([]*sim.Request{r})
	ids := a.GetCachedBlocks(r.InputTokens)
	if len(ids) != 2 || h.f.Pending() != 2 || h.f.Snapshot()["cxl"]["ready"] != 0 {
		t.Fatal("full computed blocks must be reserved, but not yet ready")
	}
	a.MirrorToCPU([]*sim.Request{r})
	if h.f.Pending() != 2 || a.Blocks[ids[0]].RefCount != 2 {
		t.Fatal("repeated observation submitted another copy/pin")
	}
	a.ReleaseKVBlocks(r)
	if a.Blocks[ids[0]].RefCount != 1 {
		t.Fatal("request completion released the DMA source pin")
	}
	for len(h.events) > 0 {
		h.next(t)
		assertPeerConservation(t, a)
	}
	if h.f.Snapshot()["cxl"]["ready"] != 2 || len(a.GetCachedBlocks(r.InputTokens)) != 2 || a.UsedBlocks() != 0 {
		t.Fatal("background completion must retain both HBM and CPU copies with no pins")
	}
	// A subsequent capacity request can discard HBM copies immediately. It
	// must not submit another victim write for copies already saved in CPU.
	if !a.makeSpace(4, nil, "next") || h.f.Pending() != 0 {
		t.Fatal("reclaim waited for another redundant write")
	}
	assertPeerConservation(t, a)
}

func TestPeerBackgroundStoreRejectsInaccessiblePool(t *testing.T) {
	h, _, _ := newPeerHarness(t, false)
	h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "missing"})
	if _, err := NewPeerCache("bad", 4, 2, h.f, nil, BuiltinPeerPolicy{Name: "lru_drop"}); err == nil {
		t.Fatal("accepted an inaccessible background destination")
	}
}
