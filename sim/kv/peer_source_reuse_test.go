package kv

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func reuseFixture(t *testing.T) (*peerHarness, *PeerCache) {
	t.Helper()
	return reuseFixtureCapacity(t, 2)
}

func reclaimReuseFixture(t *testing.T, blocks int64) (*peerHarness, *PeerCache) {
	t.Helper()
	h, s := reuseFixtureCapacity(t, blocks)
	h.f.mechanisms.BackgroundStoreMode = "on_reclaim"
	s.policy = BuiltinPeerPolicy{Name: "lfu_store"}
	return h, s
}

func TestReclaimSourceReuseNativeRunningGrantWaitsOnlyAtExecution(t *testing.T) {
	for _, decode := range []bool{false, true} {
		var observedRecords []PeerRecord
		for _, observed := range []bool{false, true} {
			t.Run(fmt.Sprintf("decode=%v/observe=%v", decode, observed), func(t *testing.T) {
				h, s := reclaimReuseFixture(t, 4)
				input, tokens, victims := 8, 4, 2
				if decode {
					input, tokens, victims = 2, 1, 1
				}
				a := capacityRequest(t, s, "active", input, 2, 100)
				a.TTFTSet = decode
				s.MirrorToCPU([]*sim.Request{a})
				seedPeer(t, s, "idle", []sim.TokenID{1, 2, 3, 4, 5, 6})
				ctx := capacityContext(s, []*sim.Request{a}, int64(tokens), 1)
				ctx.CapacityVictims = nil
				ctx.PrefillTokenThreshold = int64(tokens)
				ctx.ObserveExecutionWork = observed
				r := sim.NewVLLMNativeBatchFormation().FormBatch(ctx)
				if len(r.Preempted) != 0 || a.NumNewTokens != tokens || a.ProgressIndex != 2 || s.IsDeferred(a.ID) || len(s.storePins) != 0 {
					t.Fatal("reclaim STORE became a scheduling failure", r.Preempted, a.NumNewTokens, s.LastAllocationFailure())
				}
				deps := s.ExecutionDependencies(a.ID)
				if reclaimCount(h) != victims || len(deps) != 1 || len(deps[0].HBMBlocks) != victims || h.f.Pending() != 1 {
					t.Fatal("same-allocation grouped STORE lost its fence", reclaimCount(h), deps)
				}
				if observed && (len(r.ExecutionWork.Allocations) != 1 || !r.ExecutionWork.Allocations[0].Granted || !reflect.DeepEqual(r.ExecutionWork.Allocations[0].Dependencies, deps)) {
					t.Fatal("admitted allocation did not expose execution dependencies", r.ExecutionWork)
				}
				deps[0].HBMBlocks[0] = -1
				if s.ExecutionDependencies(a.ID)[0].HBMBlocks[0] < 0 {
					t.Fatal("dependency snapshot aliases runtime ownership")
				}
				owned := append([]int64(nil), s.RequestMap[a.ID]...)
				var end int64
				ok, err := s.BeginBatch(0, []sim.BatchWork{{Request: a, PrefixTokens: 2, NewTokens: int64(tokens)}}, func(at int64) { end = at })
				if !ok || err != nil {
					t.Fatal(ok, err)
				}
				for end == 0 {
					phaseNext(t, h)
				}
				var physical, forward, adopted int64
				for _, e := range h.records {
					switch e.Name {
					case "transfer_end":
						physical = e.Time
					case "engine_forward_ready":
						forward = e.Time
					case "transfer_adopted":
						adopted = e.Time
					}
				}
				if physical != int64(5+100*victims) || forward != physical || adopted <= forward || end != physical+20 {
					t.Fatal("physical fence/publication/compute costs conflated", physical, forward, adopted, end)
				}
				for _, id := range owned {
					if s.Blocks[id].RefCount != 1 {
						t.Fatal("old STORE completion released new owner's block")
					}
				}
				if len(s.ExecutionDependencies(a.ID)) != 0 || h.f.Pending() != 0 {
					t.Fatal("completed dependency survived")
				}
				s.ReleaseKVBlocks(a)
				assertPeerConservation(t, s)
				if observed && !reflect.DeepEqual(observedRecords, h.records) {
					t.Fatal("execution observer changed physical behavior")
				}
				observedRecords = h.records
			})
		}
	}
}

func TestReclaimSourceReuseGroupedLoadAndCancellation(t *testing.T) {
	h, s := reclaimReuseFixture(t, 2)
	seedPeer(t, s, "old", []sim.TokenID{1, 2, 3, 4})
	r := &sim.Request{ID: "restore", InputTokens: []sim.TokenID{21, 22, 23}}
	hash := s.hashes(r.InputTokens)[0]
	e, _ := h.f.reserve("cpu", hash, r.InputTokens[:2], 0)
	e.ready = true
	if s.AllocateKVBlocks(r, 0, 2, nil) || s.readPending[r.ID] != 1 || len(s.ExecutionDependencies(r.ID)) != 1 {
		t.Fatal("reclaim and LOAD in the same group lost source dependency")
	}
	s.ClearDeferred(r.ID)
	var end int64
	s.BeginBatch(0, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	var storeEnd, loadStart int64
	for _, e := range h.records {
		if e.Name == "transfer_end" && e.Destination == "cpu" {
			storeEnd = e.Time
		}
		if e.Name == "transfer_start" && e.Destination == "hbm" {
			loadStart = e.Time
		}
	}
	if storeEnd == 0 || loadStart < storeEnd {
		t.Fatal("cancelled LOAD overwrote an outstanding STORE", storeEnd, loadStart)
	}
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	end = 0
	s.BeginBatch(500, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	if h.f.Pending() != 0 || s.FreeBlockCnt != 2 || len(s.storePins) != 0 || len(s.pendingTargets) != 0 {
		t.Fatal("cancelled grouped reclaim/LOAD leaked resources")
	}
	assertPeerConservation(t, s)
}

func TestReclaimSourceReuseDoesNotTurnTrueCapacityFailureIntoSuccess(t *testing.T) {
	_, s := reclaimReuseFixture(t, 2)
	a := capacityRequest(t, s, "a", 4, 2, 100)
	capacityRequest(t, s, "holder", 2, 2, 200)
	if s.AllocateKVBlocks(a, 2, 4, nil) || s.LastAllocationFailure().Kind != "capacity" || len(s.ExecutionDependencies(a.ID)) != 0 {
		t.Fatal("active ownership was treated as recyclable STORE source")
	}
}

func TestReclaimSourceReuseReallocationBeforeAdoptionKeepsNewOwner(t *testing.T) {
	h, s := reclaimReuseFixture(t, 1)
	seedPeer(t, s, "old", []sim.TokenID{1, 2})
	r := &sim.Request{ID: "cancel", InputTokens: []sim.TokenID{11, 12}}
	if !s.AllocateKVBlocks(r, 0, 2, nil) {
		t.Fatal("initial reuse rejected")
	}
	s.ReleaseKVBlocks(r)
	var end int64
	s.BeginBatch(0, nil, func(at int64) { end = at })
	for len(s.ExecutionDependencies(r.ID)) > 0 {
		phaseNext(t, h)
	}
	if end != 0 || h.f.Pending() != 1 {
		t.Fatal("fixture missed physical-done/unadopted boundary")
	}
	fresh := &sim.Request{ID: "fresh", InputTokens: []sim.TokenID{21, 22}}
	if !s.AllocateKVBlocks(fresh, 0, 2, nil) || len(s.ExecutionDependencies(fresh.ID)) != 0 {
		t.Fatal("physically completed STORE required adoption before reuse")
	}
	for end == 0 {
		phaseNext(t, h)
	}
	id := s.RequestMap[fresh.ID][0]
	if s.Blocks[id].RefCount != 1 || s.Blocks[id].Tokens[0] != 21 {
		t.Fatal("old adoption corrupted the replacement owner")
	}
	s.ReleaseKVBlocks(fresh)
	assertPeerConservation(t, s)
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
