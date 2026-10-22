package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestRequestSpillJoinsReadyPendingAndSavesOnlyMissingCompletedBlocks(t *testing.T) {
	h, s, _ := newPeerHarness(t, false)
	r := capacityRequest(t, s, "victim", 8, 7, 10)
	s.MirrorToCPU([]*sim.Request{r})
	ids := s.RequestMap[r.ID]
	s.storeCopy(s.Blocks[ids[0]], "cxl", r.ID, true)
	h.next(t)
	s.storeCopy(s.Blocks[ids[1]], "cxl", r.ID, true)
	out := s.SpillPrefill(r, "cxl")
	if out.Status != "spill_pending" || out.ReadyBlocks != 1 || out.PendingBlocks != 1 || out.NewBlocks != 1 || out.TailTokens != 1 || h.f.Pending() != 2 {
		t.Fatal("spill did not distinguish ready/pending/missing/partial", out, h.f.Pending())
	}
	before := s.Blocks[ids[2]].RefCount
	again := s.SpillPrefill(r, "cxl")
	if again.NewBlocks != 0 || again.PendingBlocks != 2 || h.f.Pending() != 2 || s.Blocks[ids[2]].RefCount != before {
		t.Fatal("repeated spill duplicated a write/pin", again)
	}
	s.ReleaseKVBlocks(r)
	for len(h.events) > 0 {
		h.next(t)
		assertPeerConservation(t, s)
	}
	if h.f.Snapshot()["cxl"]["ready"] != 3 || h.f.Snapshot()["cxl"]["read_pins"] != 0 || s.UsedBlocks() != 0 {
		t.Fatal("release cancelled a pending write or leaked a lease")
	}
}

type refuseSpillEviction struct{}

func (refuseSpillEviction) Observe(PoolPolicyEvent)               {}
func (refuseSpillEviction) Victim([]PoolEvictionCandidate) string { return "" }

func TestRequestSpillReservationFailureRollsBackWithoutSubmitting(t *testing.T) {
	for _, protectPrefix := range []bool{false, true} {
		h, s, _ := newPeerHarness(t, false)
		r := capacityRequest(t, s, "victim", 8, 6, 10)
		s.MirrorToCPU([]*sim.Request{r})
		p := h.f.pools["cxl"]
		p.config.CapacityBlocks = 2
		if protectPrefix {
			b := s.Blocks[s.RequestMap[r.ID][0]]
			s.storeCopy(b, "cxl", r.ID, true)
			h.next(t)
		} else {
			e, _ := h.f.reserve("cxl", "unrelated", []sim.TokenID{99, 100}, 0)
			e.ready = true
			p.policy = refuseSpillEviction{}
		}
		out := s.SpillPrefill(r, "cxl")
		if out.Status != "recompute_capacity" || out.NewBlocks != 0 || h.f.Pending() != 0 || len(p.entries) != 1 {
			t.Fatal("failed spill left a reservation or transfer", out, p.entries)
		}
		if h.f.Snapshot()["cxl"]["read_pins"] != 0 || h.f.Snapshot()["cxl"]["reserved"] != 0 {
			t.Fatal("failed spill leaked target protection")
		}
		cancelled := 0
		for _, e := range h.records {
			if e.Name == "l2_reservation_cancel" {
				cancelled++
			}
		}
		if cancelled != 1 {
			t.Fatal("test did not roll back its successful partial reservation", cancelled)
		}
		s.ReleaseKVBlocks(r)
		assertPeerConservation(t, s)
		if s.UsedBlocks() != 0 {
			t.Fatal("failed spill pinned a source")
		}
	}
}

func TestRequestSpillActualPreemptionFencesReplacement(t *testing.T) {
	for _, mode := range []string{"spill", "recompute"} {
		h, s := reuseFixtureCapacity(t, 3)
		h.f.mechanisms.BackgroundStoreMode = "on_preemption"
		r := capacityRequest(t, s, "victim", 6, 4, 1)
		s.MirrorToCPU([]*sim.Request{r})
		if h.f.Pending() != 0 {
			t.Fatal("on_preemption performed an eager store")
		}
		key := s.Blocks[s.RequestMap[r.ID][0]].Hash
		fresh := &sim.Request{ID: "fresh", InputTokens: []sim.TokenID{11, 12, 13, 14, 15, 16}, State: sim.StateQueued}
		ctx := capacityContext(s, []*sim.Request{r}, 6, 1)
		ctx.CapacityVictims = nil
		ctx.PrefillPreemptions = []*sim.Request{r}
		ctx.PreemptionStorage = []sim.DecisionPreemptionStorage{{Request: r.ID, Mode: mode}}
		if mode == "spill" {
			ctx.PreemptionStorage[0].Pool = "cpu"
		}
		ctx.PrefillTokenThreshold = 6
		ctx.WaitQ.Enqueue(fresh)
		result := sim.NewBatchFormation("").FormBatch(ctx)
		if len(result.PreemptionStorage) != 1 || len(result.NewlyScheduled) != 1 || fresh.NumNewTokens != 6 {
			t.Fatal("storage action did not reach actual preemption", result)
		}
		var end int64
		if ok, err := s.BeginBatch(0, []sim.BatchWork{{Request: fresh, NewTokens: 6}}, func(at int64) { end = at }); !ok || err != nil {
			t.Fatal(ok, err)
		}
		for end == 0 {
			phaseNext(t, h)
		}
		if mode == "spill" {
			entry := h.f.pools["cpu"].entries[key]
			if end != 225 || entry == nil || !entry.ready || entry.tokens[0] != 1 || result.PreemptionStorage[0].NewBlocks != 2 {
				t.Fatal("spill source overwritten before save completed", end, result.PreemptionStorage)
			}
		} else if end != 22 || len(h.f.pools["cpu"].entries) != 0 {
			t.Fatal("recompute paid for an unrequested copy", end)
		}
		s.ReleaseKVBlocks(fresh)
		s.ReleaseKVBlocks(r)
		assertPeerConservation(t, s)
		if s.UsedBlocks() != 0 || h.f.Pending() != 0 {
			t.Fatal("spill preemption leaked resources")
		}
	}
}

func TestRequestSpillCapacityActionRunsOnlyForActualVictim(t *testing.T) {
	for _, capacity := range []int64{4, 8} {
		h, s, _ := newPeerHarness(t, false)
		s.KVCacheState = NewKVCacheState(capacity, 2)
		s.policy = BuiltinPeerPolicy{Name: "lru_drop"}
		a := capacityRequest(t, s, "a", 8, 2, 10)
		b := capacityRequest(t, s, "b", 8, 2, 30)
		d := capacityRequest(t, s, "decode", 1, 1, 90)
		d.TTFTSet = true
		s.MirrorToCPU([]*sim.Request{a, b, d})
		ctx := capacityContext(s, []*sim.Request{a, b, d}, 4, 3, "a")
		ctx.PreemptionStorage = []sim.DecisionPreemptionStorage{{Request: "a", Mode: "spill", Pool: "cxl"}}
		result := sim.NewBatchFormation("").FormBatch(ctx)
		if capacity == 4 {
			if len(result.Preempted) != 1 || len(result.PreemptionStorage) != 1 || result.PreemptionStorage[0].NewBlocks != 1 || h.f.Pending() != 1 {
				t.Fatal("actual capacity victim did not spill its completed prefix", result)
			}
			if result.Capacity[0].Failure.Kind != "capacity" || result.Capacity[0].Victim != "a" || b.NumNewTokens != 2 || d.NumNewTokens != 1 {
				t.Fatal("spill changed victim/grant/decode semantics", result.Capacity)
			}
		} else if len(result.Preempted) != 0 || len(result.PreemptionStorage) != 0 || h.f.Pending() != 0 {
			t.Fatal("a conditional preference submitted an unnecessary spill")
		}
		for _, r := range []*sim.Request{a, b, d} {
			s.ReleaseKVBlocks(r)
		}
		for len(h.events) > 0 {
			h.next(t)
		}
		assertPeerConservation(t, s)
		if s.UsedBlocks() != 0 || h.f.Snapshot()["cxl"]["read_pins"] != 0 {
			t.Fatal("capacity spill leaked resources")
		}
	}
}

func TestRequestSpillPreservesSharedSourceAndAllReadyCopies(t *testing.T) {
	h, s, _ := newPeerHarness(t, false)
	a := capacityRequest(t, s, "a", 8, 4, 10)
	s.MirrorToCPU([]*sim.Request{a})
	b := &sim.Request{ID: "b", State: sim.StateRunning, InputTokens: append([]sim.TokenID(nil), a.InputTokens...)}
	b.InputTokens[4] = 99
	ids := s.GetCachedBlocks(b.InputTokens)
	if len(ids) != 2 || !s.AllocateKVBlocks(b, 4, 5, ids) {
		t.Fatal("shared source fixture failed")
	}
	b.ProgressIndex = 5
	ctx := capacityContext(s, []*sim.Request{a, b}, 0, 2)
	ctx.CapacityVictims = nil
	ctx.PrefillPreemptions = []*sim.Request{a}
	ctx.PreemptionStorage = []sim.DecisionPreemptionStorage{{Request: "a", Mode: "spill", Pool: "cxl"}}
	result := sim.NewBatchFormation("").FormBatch(ctx)
	if len(result.PreemptionStorage) != 1 || result.PreemptionStorage[0].NewBlocks != 2 {
		t.Fatal("missing shared spill")
	}
	for len(h.events) > 0 {
		h.next(t)
	}
	for _, id := range ids {
		if s.Blocks[id].RefCount != 1 {
			t.Fatal("spill completion released surviving owner")
		}
	}
	ready := s.SpillPrefill(b, "cxl")
	if ready.Status != "spill_ready" || ready.ReadyBlocks != 2 || ready.NewBlocks != 0 || ready.TailTokens != 1 || h.f.Pending() != 0 {
		t.Fatal("all-ready spill duplicated an existing shared copy", ready)
	}
	s.ReleaseKVBlocks(b)
	assertPeerConservation(t, s)
	if s.UsedBlocks() != 0 || h.f.Snapshot()["cxl"]["read_pins"] != 0 {
		t.Fatal("shared spill leaked ownership")
	}
}

func TestRequestSpillTouchesExistingPrefixAfterAtomicReservation(t *testing.T) {
	h, s, _ := newPeerHarness(t, false)
	if err := h.f.SetPoolEvictionPolicy("cxl", &PrefixLRU{}); err != nil {
		t.Fatal(err)
	}
	r := capacityRequest(t, s, "victim", 8, 4, 10)
	s.MirrorToCPU([]*sim.Request{r})
	first := s.Blocks[s.RequestMap[r.ID][0]]
	s.storeCopy(first, "cxl", r.ID, true)
	h.next(t)
	other, _ := h.f.reserve("cxl", "other", []sim.TokenID{90, 91}, s.clock)
	other.ready = true
	h.f.poolPolicyEvent("cxl", "ready", "other", []string{"other"}, s.clock)
	if h.f.pools["cxl"].evictionVictim().hash != first.Hash {
		t.Fatal("fixture needs an older prefix")
	}
	out := s.SpillPrefill(r, "cxl")
	if out.NewBlocks != 1 || out.ReadyBlocks != 1 || h.f.pools["cxl"].evictionVictim().hash != "other" {
		t.Fatal("successful native-style spill failed to touch existing prefix", out)
	}
	s.ReleaseKVBlocks(r)
	for len(h.events) > 0 {
		h.next(t)
	}
	assertPeerConservation(t, s)
}
