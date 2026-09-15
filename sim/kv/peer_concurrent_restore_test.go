package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func concurrentRestoreFixture(t *testing.T, blocks int64) (*peerHarness, *PeerCache, [][]sim.TokenID) {
	t.Helper()
	h, source, target := newPeerHarness(t, false)
	target.KVCacheState = NewKVCacheState(blocks, 2)
	target.policy = BuiltinPeerPolicy{Name: "lru_drop"}
	tokens := [][]sim.TokenID{{10, 11, 12, 13, 14}, {20, 21, 22, 23, 24}}
	for i, toks := range tokens {
		r := seedPeer(t, source, string(rune('a'+i)), toks)
		for _, id := range source.GetCachedBlocks(r.InputTokens) {
			if !source.store(source.Blocks[id], "cxl", "save") {
				t.Fatal("fixture store rejected")
			}
		}
		for len(h.events) > 0 {
			h.next(t)
		}
	}
	if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "cxl", GroupTransfers: true, ConcurrentRestores: true, RestoreWindow: 16}); err != nil {
		t.Fatal(err)
	}
	return h, target, tokens
}

func TestConcurrentRestoresReserveAndPublishIndependently(t *testing.T) {
	h, target, tokens := concurrentRestoreFixture(t, 6)
	a := &sim.Request{ID: "a", InputTokens: tokens[0]}
	b := &sim.Request{ID: "b", InputTokens: tokens[1]}
	for _, r := range []*sim.Request{a, b} {
		if target.AllocateKVBlocks(r, 0, 2, nil) || !target.IsDeferred(r.ID) {
			t.Fatal("restore was not deferred")
		}
	}
	if h.f.Pending() != 2 || target.FreeBlockCnt != 2 || h.f.Snapshot()["cxl"]["read_pins"] != 4 {
		t.Fatal("second restore did not reserve its complete range")
	}
	// Repeated decisions must not duplicate reads or reservations.
	if target.AllocateKVBlocks(a, 0, 2, nil) || h.f.Pending() != 2 {
		t.Fatal("repeated decision submitted duplicate work")
	}
	if len(target.GetCachedBlocks(a.InputTokens)) != 0 || len(target.GetCachedBlocks(b.InputTokens)) != 0 {
		t.Fatal("pending targets visible as prefix hits")
	}
	h.next(t)
	if len(target.GetCachedBlocks(a.InputTokens)) != 2 || len(target.GetCachedBlocks(b.InputTokens)) != 0 {
		t.Fatal("completion published another request's pending targets")
	}
	if !target.AllocateKVBlocks(a, 4, 5, target.GetCachedBlocks(a.InputTokens)) {
		t.Fatal("first request cannot consume completed restore")
	}
	target.ReleaseKVBlocks(a)
	h.next(t)
	if !target.AllocateKVBlocks(b, 4, 5, target.GetCachedBlocks(b.InputTokens)) {
		t.Fatal("second request cannot consume completed restore")
	}
	target.ReleaseKVBlocks(b)
	assertPeerConservation(t, target)
	if target.FreeBlockCnt != 6 || len(target.holds) != 0 || len(target.prefillTargets) != 0 || h.f.Pending() != 0 {
		t.Fatal("restore lifecycle leaked reservations")
	}
}

func TestConcurrentRestoreLeavesOtherPrefillHeadroom(t *testing.T) {
	h, target, tokens := concurrentRestoreFixture(t, 5)
	running := &sim.Request{ID: "compute", InputTokens: []sim.TokenID{30, 31, 32, 33, 34, 35, 36}}
	if !target.AllocateKVBlocks(running, 0, 2, nil) {
		t.Fatal("first prefill chunk rejected")
	}
	reader := &sim.Request{ID: "reader", InputTokens: tokens[0]}
	before := target.FreeBlockCnt
	if target.AllocateKVBlocks(reader, 0, 2, nil) || target.IsDeferred(reader.ID) || h.f.Pending() != 0 || target.FreeBlockCnt != before {
		t.Fatal("load consumed the running prefill's three remaining blocks")
	}
	// Compute can still use its reservation. Once prefill ends, the reservation
	// disappears; releasing the request then permits a real asynchronous load.
	if !target.AllocateKVBlocks(running, 2, 7, nil) {
		t.Fatal("reservation did not protect forward progress")
	}
	target.ReleaseKVBlocks(running)
	if target.AllocateKVBlocks(reader, 0, 2, nil) || h.f.Pending() != 1 {
		t.Fatal("released capacity did not admit the pending candidate")
	}
	h.next(t)
	if !target.AllocateKVBlocks(reader, 4, 5, target.GetCachedBlocks(reader.InputTokens)) {
		t.Fatal("restore consumer blocked")
	}
	target.ReleaseKVBlocks(reader)
	assertPeerConservation(t, target)
}

func TestConcurrentRestoreRejectsPartialCapacityReservation(t *testing.T) {
	h, target, tokens := concurrentRestoreFixture(t, 4)
	busy := &sim.Request{ID: "busy", InputTokens: []sim.TokenID{30, 31, 32}}
	if !target.AllocateKVBlocks(busy, 0, 3, nil) {
		t.Fatal("fixture admission")
	}
	reader := &sim.Request{ID: "reader", InputTokens: tokens[0]}
	if target.AllocateKVBlocks(reader, 0, 2, nil) || h.f.Pending() != 0 || target.FreeBlockCnt != 2 {
		t.Fatal("load fit, but full prompt did not: must reject without partial work")
	}
	target.ReleaseKVBlocks(busy)
	assertPeerConservation(t, target)
}

func TestConcurrentRestoreCancellationKeepsDMAPinsUntilCompletion(t *testing.T) {
	h, target, tokens := concurrentRestoreFixture(t, 6)
	reader := &sim.Request{ID: "cancelled", InputTokens: tokens[0]}
	if target.AllocateKVBlocks(reader, 0, 2, nil) {
		t.Fatal("load admitted early")
	}
	target.ClearDeferred(reader.ID)
	target.ClearDeferred(reader.ID)
	if target.FreeBlockCnt != 4 || len(target.prefillTargets) != 0 || len(target.ReadyDeferredRequests()) != 0 {
		t.Fatal("cancellation freed an active DMA target or retained admission state")
	}
	h.next(t)
	if target.FreeBlockCnt != 6 || len(target.holds) != 0 || len(target.cancelledRestores) != 0 || h.f.Snapshot()["cxl"]["read_pins"] != 0 {
		t.Fatal("late completion recreated cancelled request's pins")
	}
	assertPeerConservation(t, target)
}
