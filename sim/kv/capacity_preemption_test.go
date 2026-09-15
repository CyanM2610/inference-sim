package kv

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func capacityRequest(t *testing.T, store sim.KVStore, id string, input, computed, base int) *sim.Request {
	t.Helper()
	r := &sim.Request{ID: id, State: sim.StateRunning, OutputTokens: []sim.TokenID{9, 10}}
	for i := 0; i < input; i++ {
		r.InputTokens = append(r.InputTokens, sim.TokenID(base+i))
	}
	if !store.AllocateKVBlocks(r, 0, int64(computed), nil) {
		t.Fatal("cannot construct resident request", id)
	}
	r.ProgressIndex = int64(computed)
	return r
}

func capacityContext(store sim.KVStore, running []*sim.Request, tokens, slots int64, order ...string) sim.BatchContext {
	computed := map[string]int64{}
	for _, r := range running {
		computed[r.ID] = r.ProgressIndex
	}
	return sim.BatchContext{KVCache: store, RunningBatch: &sim.Batch{Requests: running}, WaitQ: &sim.WaitQueue{},
		MaxNumBatchedTokens: tokens, MaxNumSeqs: slots, PrefillTokenThreshold: 2, ComputedTokens: computed,
		CapacityVictims: &sim.DecisionCapacityVictims{Order: order}}
}

func TestCapacityVictimsDoNothingWithoutHBMPressure(t *testing.T) {
	store := NewKVCacheState(12, 2)
	a := capacityRequest(t, store, "a", 8, 2, 10)
	b := capacityRequest(t, store, "b", 8, 2, 30)
	ctx := capacityContext(store, []*sim.Request{a, b}, 4, 2, "a", "b")
	ctx.WaitQ.Enqueue(&sim.Request{ID: "waiting", InputTokens: []sim.TokenID{90, 91}})
	r := sim.NewBatchFormation("").FormBatch(ctx)
	if len(r.Preempted) != 0 || len(r.Capacity) != 0 || a.NumNewTokens != 2 || b.NumNewTokens != 2 {
		t.Fatal("full sequence slots triggered a pressure victim", r)
	}
}

func TestCapacityVictimRefundsGrantedTokensAndKeepsDecode(t *testing.T) {
	store := NewKVCacheState(4, 2)
	a := capacityRequest(t, store, "a", 8, 2, 10)
	b := capacityRequest(t, store, "b", 8, 2, 30)
	d := capacityRequest(t, store, "decode", 1, 1, 90)
	d.TTFTSet = true
	ctx := capacityContext(store, []*sim.Request{a, b, d}, 4, 3, "a")
	r := sim.NewBatchFormation("").FormBatch(ctx)
	if len(r.Preempted) != 1 || r.Preempted[0].Request != a || r.Preempted[0].ComputedTokensBefore != 2 || r.Preempted[0].Reason != "policy_capacity_prefill" {
		t.Fatal("wrong actual pressure victim", r.Preempted)
	}
	if a.State != sim.StateQueued || a.NumNewTokens != 0 || b.NumNewTokens != 2 || d.NumNewTokens != 1 || d.State != sim.StateRunning {
		t.Fatal("grant refund/index adjustment/decode protection failed", a.NumNewTokens, b.NumNewTokens, d.NumNewTokens)
	}
	if _, exists := ctx.ComputedTokens[a.ID]; exists {
		t.Fatal("preempted speculative progress survived")
	}
	if len(r.Capacity) != 1 || r.Capacity[0].Failure.Kind != "capacity" || r.Capacity[0].Victim != "a" {
		t.Fatal("missing causal feedback", r.Capacity)
	}
}

func TestCapacityPressureRetriesAfterSharedReferences(t *testing.T) {
	store := NewKVCacheState(4, 2)
	a := capacityRequest(t, store, "a", 12, 4, 10)
	b := &sim.Request{ID: "b", State: sim.StateRunning, InputTokens: append([]sim.TokenID(nil), a.InputTokens...), OutputTokens: []sim.TokenID{9}}
	// Claim the two shared blocks and compute into a third block, then limit
	// the tight request to the final block. Releasing A alone leaves B's refs.
	if !store.AllocateKVBlocks(b, 4, 5, store.GetCachedBlocks(b.InputTokens)) {
		t.Fatal("shared fixture allocation failed")
	}
	b.ProgressIndex = 5
	tight := capacityRequest(t, store, "tight", 8, 2, 50)
	ctx := capacityContext(store, []*sim.Request{tight, a, b}, 4, 3, "a", "b")
	ctx.PrefillTokenThreshold = 4
	r := sim.NewBatchFormation("").FormBatch(ctx)
	var victims []string
	for _, p := range r.Preempted {
		victims = append(victims, p.Request.ID)
	}
	if !reflect.DeepEqual(victims, []string{"a", "b"}) || tight.NumNewTokens != 4 || len(r.Capacity) != 2 {
		t.Fatal("logical release was mistaken for physical free space", victims, tight.NumNewTokens, r.Capacity)
	}
}

func TestCapacityPressureAtWaitingAdmission(t *testing.T) {
	store := NewKVCacheState(2, 2)
	a := capacityRequest(t, store, "a", 8, 2, 10)
	ctx := capacityContext(store, []*sim.Request{a}, 4, 2, "a")
	w := &sim.Request{ID: "waiting", State: sim.StateQueued, InputTokens: []sim.TokenID{90, 91, 92, 93}, OutputTokens: []sim.TokenID{9}}
	ctx.WaitQ.Enqueue(w)
	r := sim.NewBatchFormation("").FormBatch(ctx)
	if len(r.Preempted) != 1 || len(r.NewlyScheduled) != 1 || r.NewlyScheduled[0].Request != w || w.NumNewTokens != 2 || a.NumNewTokens != 0 || a.State != sim.StateQueued {
		t.Fatal("waiting pressure was not served through the same runtime", r)
	}
	if r.Capacity[0].Failure.Request != w.ID || ctx.WaitQ.Len() != 1 || ctx.WaitQ.Peek() != a {
		t.Fatal("yielded victim immediately reentered or feedback lost requester")
	}
}

func TestCapacityNoVictimStillRunsAnUnblockedDecode(t *testing.T) {
	store := NewKVCacheState(2, 2)
	a := capacityRequest(t, store, "protected", 8, 2, 10)
	d := capacityRequest(t, store, "decode", 1, 1, 90)
	d.TTFTSet = true
	ctx := capacityContext(store, []*sim.Request{a, d}, 4, 2)
	r := sim.NewBatchFormation("").FormBatch(ctx)
	if len(r.Preempted) != 0 || a.NumNewTokens != 0 || d.NumNewTokens != 1 || r.Capacity[0].Status != "no_eligible_victim" {
		t.Fatal("a protected request blocked independent decode progress", r)
	}
}

func TestCapacityVictimsWaitForExistingStoreInsteadOfPreempting(t *testing.T) {
	h, store, _ := newPeerHarness(t, false)
	a := capacityRequest(t, store, "active", 8, 2, 100)
	seedPeer(t, store, "idle", []sim.TokenID{1, 2, 3, 4, 5, 6})
	for _, b := range store.Blocks {
		store.frequency[b.Hash] = 1
	}
	ctx := capacityContext(store, []*sim.Request{a}, 2, 1, "active")
	for i := 0; i < 4; i++ {
		r := sim.NewBatchFormation("").FormBatch(ctx)
		if len(r.Preempted) != 0 || reclaimCount(h) != 1 || a.NumNewTokens != 0 {
			t.Fatal("waiting for STORE appended a request/block victim", i, len(r.Preempted), reclaimCount(h))
		}
		if len(r.Capacity) != 1 || r.Capacity[0].Failure.Kind != "wait" || r.Capacity[0].Failure.Reason != "reclaim_writeback" {
			t.Fatal("STORE promise not reported as waiting", r.Capacity)
		}
	}
	for len(h.events) > 0 {
		h.next(t)
	}
	r := sim.NewBatchFormation("").FormBatch(ctx)
	if len(r.Preempted) != 0 || a.NumNewTokens != 2 || store.LastAllocationFailure().Kind != "" {
		t.Fatal("adopted STORE failed to unblock original computation")
	}
	assertPeerConservation(t, store)
}
