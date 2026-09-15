package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

func TestPendingRestoreDoesNotBlockResidentRequest(t *testing.T) {
	h, source, target := newPeerHarness(t, false)
	warm := seedPeer(t, target, "warm", []sim.TokenID{10, 11, 12})
	remote := seedPeer(t, source, "remote", []sim.TokenID{20, 21, 22})
	block := source.GetCachedBlocks(remote.InputTokens)[0]
	if !source.store(source.Blocks[block], "dram", "save") {
		t.Fatal("store failed")
	}
	h.next(t)
	reader := &sim.Request{ID: "loading", InputTokens: remote.InputTokens}
	if target.AllocateKVBlocks(reader, 0, 3, nil) {
		t.Fatal("load became ready early")
	}
	if h.f.Pending() != 1 {
		t.Fatal("missing pending load")
	}
	resident := &sim.Request{ID: "resident", InputTokens: warm.InputTokens}
	cached := target.GetCachedBlocks(resident.InputTokens)
	if len(cached) != 1 {
		t.Fatal("fixture lost resident prefix")
	}
	if !target.AllocateKVBlocks(resident, 2, 3, cached) {
		t.Fatal("resident request blocked by an unrelated restore")
	}
	if h.f.Pending() != 1 || len(target.GetCachedBlocks(reader.InputTokens)) != 0 {
		t.Fatal("resident admission altered the unfinished transfer")
	}
	assertPeerConservation(t, target)
	target.ReleaseKVBlocks(resident)
	h.next(t)
	cached = target.GetCachedBlocks(reader.InputTokens)
	if len(cached) != 1 || !target.AllocateKVBlocks(reader, 2, 3, cached) {
		t.Fatal("loading request cannot consume completed transfer")
	}
	target.ReleaseKVBlocks(reader)
	assertPeerConservation(t, target)
}

func TestPendingRestoreProtectsItsExistingLocalPrefix(t *testing.T) {
	h, source, target := newPeerHarness(t, false)
	target.policy = BuiltinPeerPolicy{Name: "lru_drop"}
	seedPeer(t, target, "local", []sim.TokenID{10, 11, 99})
	remote := seedPeer(t, source, "remote", []sim.TokenID{10, 11, 20, 21, 22})
	blocks := source.GetCachedBlocks(remote.InputTokens)
	if !source.store(source.Blocks[blocks[1]], "dram", "save") {
		t.Fatal("store failed")
	}
	h.next(t)
	reader := &sim.Request{ID: "loading", InputTokens: remote.InputTokens}
	cached := target.GetCachedBlocks(reader.InputTokens)
	if len(cached) != 1 {
		t.Fatal("fixture lacks partial local prefix")
	}
	local := cached[0]
	if target.AllocateKVBlocks(reader, 2, 5, cached) {
		t.Fatal("early admission")
	}
	if target.Blocks[local].RefCount != 1 {
		t.Fatal("local prefix not protected during restore")
	}
	other := &sim.Request{ID: "cold", InputTokens: []sim.TokenID{30, 31, 32}}
	if !target.AllocateKVBlocks(other, 0, 3, nil) {
		t.Fatal("independent compute cannot use spare capacity")
	}
	if len(target.GetCachedBlocks(reader.InputTokens)) != 1 {
		t.Fatal("other request evicted pending owner's local prefix")
	}
	target.ReleaseKVBlocks(other)
	h.next(t)
	// A finished restore must get a chance to consume its reservation.
	if target.AllocateKVBlocks(other, 0, 3, nil) {
		t.Fatal("new admission bypassed the ready restore owner")
	}
	cached = target.GetCachedBlocks(reader.InputTokens)
	if len(cached) != 2 || !target.AllocateKVBlocks(reader, 4, 5, cached) {
		t.Fatal("completed prefix not usable")
	}
	target.ReleaseKVBlocks(reader)
	if target.PeerSnapshot()["active_or_pinned"] != 0 {
		t.Fatal("prefix pins leaked")
	}
	assertPeerConservation(t, target)
}
