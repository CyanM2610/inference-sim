package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

func TestExplicitPrefillPreemptionPreservesDeferredStoreSourceFence(t *testing.T) {
	h, cache := reuseFixtureCapacity(t, 3)
	old := &sim.Request{ID: "old", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6}, State: sim.StateRunning}
	if !cache.AllocateKVBlocks(old, 0, 4, nil) {
		t.Fatal("old allocation failed")
	}
	old.ProgressIndex = 4
	old.NumNewTokens = 4
	cache.MirrorToCPU([]*sim.Request{old})
	oldHash := cache.Blocks[cache.RequestMap[old.ID][0]].Hash
	fresh := &sim.Request{ID: "fresh", InputTokens: []sim.TokenID{11, 12, 13, 14}, State: sim.StateQueued}
	wait := &sim.WaitQueue{}
	wait.Enqueue(fresh)
	computed := map[string]int64{old.ID: 4}
	ctx := sim.BatchContext{RunningBatch: &sim.Batch{Requests: []*sim.Request{old}}, WaitQ: wait, KVCache: cache, MaxNumBatchedTokens: 4, MaxNumSeqs: 1, ComputedTokens: computed, PrefillPreemptions: []*sim.Request{old}}
	result := sim.NewBatchFormation("").FormBatch(ctx)
	if len(result.Preempted) != 1 || result.PreemptionHappened || len(result.NewlyScheduled) != 1 || result.NewlyScheduled[0].Request != fresh {
		t.Fatal("explicit victim blocked its replacement", result)
	}
	if old.State != sim.StateQueued || old.ProgressIndex != 0 || old.NumNewTokens != 0 || computed[old.ID] != 0 || wait.Peek() != old {
		t.Fatal("victim progress/requeue was not reset")
	}
	if h.f.Pending() != 1 {
		t.Fatal("preemption cancelled or duplicated the old store", h.f.Pending())
	}
	newIDs := append([]int64(nil), cache.RequestMap[fresh.ID]...)
	var end int64
	if ok, err := cache.BeginBatch(0, []sim.BatchWork{{Request: fresh, NewTokens: 4}}, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	if end != 225 {
		t.Fatal("replacement overwrote a source before store completion", end)
	}
	entry := h.f.pools["cpu"].entries[oldHash]
	if entry == nil || !entry.ready || entry.tokens[0] != 1 {
		t.Fatal("old bytes lost before publication")
	}
	for _, id := range newIDs {
		if cache.Blocks[id].RefCount != 1 {
			t.Fatal("store completion altered replacement refs", id)
		}
	}
	cache.ReleaseKVBlocks(fresh)
	assertPeerConservation(t, cache)
	if cache.FreeBlockCnt != cache.TotalCapacity() || h.f.Pending() != 0 {
		t.Fatal("preemption leaked pins/reservations")
	}
}

func TestExplicitPrefillPreemptionKeepsOtherRequestSharedReferences(t *testing.T) {
	h, cache := reuseFixtureCapacity(t, 3)
	h.f.mechanisms.BackgroundStorePool = "" // This case isolates shared request ownership.
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6}, State: sim.StateRunning}
	b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{1, 2, 3, 4, 15, 16}, State: sim.StateRunning}
	if !cache.AllocateKVBlocks(a, 0, 4, nil) {
		t.Fatal("initial allocation failed")
	}
	a.ProgressIndex = 4
	cache.MirrorToCPU([]*sim.Request{a})
	if !cache.AllocateKVBlocks(b, 4, 5, cache.GetCachedBlocks(b.FullInputTokens())) {
		t.Fatal("shared allocation failed")
	}
	b.ProgressIndex = 5
	ids := append([]int64(nil), cache.RequestMap[b.ID]...)
	if len(ids) != 3 {
		t.Fatal("test did not establish two shared and one private block")
	}
	for _, id := range ids[:2] {
		if cache.Blocks[id].RefCount != 2 {
			t.Fatal("test did not establish shared references")
		}
	}
	wait := &sim.WaitQueue{}
	result := sim.NewBatchFormation("").FormBatch(sim.BatchContext{RunningBatch: &sim.Batch{Requests: []*sim.Request{a, b}}, WaitQ: wait, KVCache: cache, MaxNumBatchedTokens: 0, MaxNumSeqs: 2, ComputedTokens: map[string]int64{a.ID: 4, b.ID: 5}, PrefillPreemptions: []*sim.Request{a}})
	if len(result.RunningBatch.Requests) != 1 || result.RunningBatch.Requests[0] != b || cache.FreeBlockCnt != cache.TotalCapacity()-int64(len(ids)) {
		t.Fatal("shared ownership incorrectly treated as free capacity")
	}
	for _, id := range ids {
		if cache.Blocks[id].RefCount != 1 {
			t.Fatal("other request lost its shared reference")
		}
	}
	cache.ReleaseKVBlocks(b)
	assertPeerConservation(t, cache)
	if cache.FreeBlockCnt != cache.TotalCapacity() {
		t.Fatal("shared references leaked")
	}
}

func TestExplicitPrefillPreemptionFlushesStoreWithoutSourceOverwrite(t *testing.T) {
	h, cache := reuseFixtureCapacity(t, 6)
	old := &sim.Request{ID: "old", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6}, State: sim.StateRunning}
	if !cache.AllocateKVBlocks(old, 0, 4, nil) {
		t.Fatal("old allocation failed")
	}
	old.ProgressIndex = 4
	cache.MirrorToCPU([]*sim.Request{old})
	short := &sim.Request{ID: "short", InputTokens: []sim.TokenID{1, 2, 3, 4, 10}, State: sim.StateQueued}
	wait := &sim.WaitQueue{}
	wait.Enqueue(short)
	result := sim.NewBatchFormation("").FormBatch(sim.BatchContext{RunningBatch: &sim.Batch{Requests: []*sim.Request{old}}, WaitQ: wait, KVCache: cache, MaxNumBatchedTokens: 4, MaxNumSeqs: 1, ComputedTokens: map[string]int64{old.ID: 4}, PrefillPreemptions: []*sim.Request{old}})
	if len(result.NewlyScheduled) != 1 || short.NumNewTokens != 1 {
		t.Fatal("replacement did not use shared prefix")
	}
	var end int64
	cache.BeginBatch(0, []sim.BatchWork{{Request: short, PrefixTokens: 4, NewTokens: 1}}, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	dependencies := 0
	for _, e := range h.records {
		if e.Name == "hbm_reuse_dependency" {
			t.Fatal("test overwrote a store source")
		}
		if e.Name == "preemption_store_dependency" {
			dependencies++
		}
	}
	if end != 225 || dependencies != 1 {
		t.Fatal("request preemption failed to flush an unmodified source", end, dependencies)
	}
	cache.ReleaseKVBlocks(short)
	assertPeerConservation(t, cache)
}
