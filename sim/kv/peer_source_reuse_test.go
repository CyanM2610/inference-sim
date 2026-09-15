package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func reuseFixture(t *testing.T) (*peerHarness, *PeerCache) {
	t.Helper()
	return reuseFixtureCapacity(t, 2)
}

func reuseFixtureCapacity(t *testing.T, capacity int64) (*peerHarness, *PeerCache) {
	t.Helper()
	h := &peerHarness{}
	var err error
	h.f, err = NewPeerFabric([]PeerResource{{ID: "copy", BytesPerUS: 1}}, []PeerPoolConfig{{ID: "cpu", CapacityBlocks: 8}}, 100, func(r PeerRecord) { h.records = append(h.records, r) })
	if err != nil {
		t.Fatal(err)
	}
	if err = h.f.ConfigureMechanisms(PeerMechanisms{GroupTransfers: true, ConcurrentRestores: true, BackgroundStorePool: "cpu", RestoreWindow: 8}); err != nil {
		t.Fatal(err)
	}
	s, err := NewPeerCache("one", capacity, 2, h.f, []PeerAccess{{Pool: "cpu", ReadPath: []string{"copy"}, WritePath: []string{"copy"}}}, BuiltinPeerPolicy{Name: "lru_drop"})
	if err != nil {
		t.Fatal(err)
	}
	s.BindEvents(func(e sim.Event) { h.events = append(h.events, e) }, func(int64) {})
	if err = s.ConfigureEnginePhases(fixedEnginePhases{EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 20, OutputReadyUS: 20, PollUS: 1, TailUS: 2}, 3}, nil); err != nil {
		t.Fatal(err)
	}
	if err = s.EnableStoreSourceReuse(); err != nil {
		t.Fatal(err)
	}
	return h, s
}

func TestStoreSourceReuseFencesOverwriteWithoutOwningNewAllocation(t *testing.T) {
	h, s := reuseFixture(t)
	old := &sim.Request{ID: "old", InputTokens: []sim.TokenID{1, 2, 3, 4}}
	if !s.AllocateKVBlocks(old, 0, 4, nil) {
		t.Fatal("old allocation failed")
	}
	old.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{old})
	oldHash := s.Blocks[s.RequestMap[old.ID][0]].Hash
	s.ReleaseKVBlocks(old)
	if s.FreeBlockCnt != 2 {
		t.Fatal("pending STORE still owns allocator refs", s.FreeBlockCnt)
	}
	if h.f.Pending() != 1 {
		t.Fatal("two blocks should remain one grouped STORE")
	}
	fresh := &sim.Request{ID: "fresh", InputTokens: []sim.TokenID{11, 12, 13, 14}}
	if !s.AllocateKVBlocks(fresh, 0, 4, nil) {
		t.Fatal("copy lease blocked new request admission")
	}
	newIDs := append([]int64(nil), s.RequestMap[fresh.ID]...)
	newHash := s.Blocks[newIDs[0]].Hash
	var end int64
	if ok, err := s.BeginBatch(0, []sim.BatchWork{{Request: fresh, NewTokens: 4}}, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	// Grouped STORE: pre=2, host=3, two 100-byte blocks=200; forward waits
	// until 205. Compute/output=20 plus the 203-us delay, tail=2 => 225.
	if end != 225 {
		t.Fatal("forward did not wait for old source content", end)
	}
	if e := h.f.pools["cpu"].entries[oldHash]; e == nil || !e.ready || e.tokens[0] != 1 {
		t.Fatal("old contents were replaced before publication")
	}
	for _, id := range newIDs {
		if s.Blocks[id].RefCount != 1 {
			t.Fatal("old completion changed new owner's refs", id, s.Blocks[id].RefCount)
		}
	}
	if s.Blocks[newIDs[0]].Hash != newHash {
		t.Fatal("old completion invalidated new content")
	}
	s.ReleaseKVBlocks(fresh)
	assertPeerConservation(t, s)
	if s.FreeBlockCnt != 2 || h.f.Pending() != 0 {
		t.Fatal("reuse leaked resources")
	}
}

func TestStoreSourceReuseOnlyWaitsForReallocatedSource(t *testing.T) {
	h, s := reuseFixture(t)
	seedPeer(t, s, "a", []sim.TokenID{1, 2})
	seedPeer(t, s, "b", []sim.TokenID{11, 12})
	r := &sim.Request{ID: "fresh", InputTokens: []sim.TokenID{21, 22}}
	if !s.AllocateKVBlocks(r, 0, 2, nil) {
		t.Fatal("new allocation failed")
	}
	var end int64
	s.BeginBatch(0, []sim.BatchWork{{Request: r, NewTokens: 2}}, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	// A completes at 105; B's store remains in flight until 205. Waiting for
	// all outstanding stores would unnecessarily push completion to 225.
	if end != 125 {
		t.Fatal("waited for an unrelated source", end)
	}
	if h.f.Pending() != 1 {
		t.Fatal("unrelated transfer should still be pending")
	}
	s.ReleaseKVBlocks(r)
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	end = 0
	s.BeginBatch(300, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	assertPeerConservation(t, s)
}

func TestStoreSourceReuseHitDoesNotRequireOverwriteFence(t *testing.T) {
	h, s := reuseFixture(t)
	old := seedPeer(t, s, "old", []sim.TokenID{1, 2, 3})
	// One full cached block is read by a new request; the other physical block
	// receives its suffix. Reading a store source must not wait for the STORE.
	r := &sim.Request{ID: "hit", InputTokens: old.InputTokens}
	if !s.AllocateKVBlocks(r, 2, 3, s.GetCachedBlocks(r.InputTokens)) {
		t.Fatal("hit rejected")
	}
	var end int64
	s.BeginBatch(0, []sim.BatchWork{{Request: r, PrefixTokens: 2, NewTokens: 1}}, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	if end != 25 {
		t.Fatal("read-only hit fenced an outstanding store", end)
	}
	s.ReleaseKVBlocks(r)
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	end = 0
	s.BeginBatch(300, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	assertPeerConservation(t, s)
}

func TestStoreSourceReuseLoadTargetWaitsAndSurvivesCancellation(t *testing.T) {
	h, s := reuseFixture(t)
	seedPeer(t, s, "old", []sim.TokenID{1, 2, 3, 4})
	r := &sim.Request{ID: "load", InputTokens: []sim.TokenID{21, 22, 23}}
	hash := s.hashes(r.InputTokens)[0]
	e, fresh := h.f.reserve("cpu", hash, r.InputTokens[:2], 0)
	if !fresh {
		t.Fatal("fixture CPU allocation failed")
	}
	e.ready = true
	if s.AllocateKVBlocks(r, 0, 2, nil) || s.readPending[r.ID] != 1 {
		t.Fatal("load did not reserve reused source block")
	}
	s.ClearDeferred(r.ID) // cancellation cannot drop either content dependency
	var end int64
	s.BeginBatch(0, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	var storeEnd, loadStart int64
	for _, row := range h.records {
		if row.Name == "transfer_end" && row.Destination == "cpu" {
			storeEnd = row.Time
		}
		if row.Name == "transfer_start" && row.Destination == "hbm" {
			loadStart = row.Time
		}
	}
	if storeEnd == 0 || loadStart < storeEnd {
		t.Fatal("H2D overwrote a live STORE source", storeEnd, loadStart)
	}
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	end = 0
	s.BeginBatch(500, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	if s.FreeBlockCnt != 2 || h.f.Pending() != 0 || len(s.holds) != 0 {
		t.Fatal("cancelled reused load leaked ownership")
	}
	assertPeerConservation(t, s)
}
