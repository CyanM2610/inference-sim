package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestHostLookupOwnsSourcesWhileHBMAdmissionWaits(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	busy := &sim.Request{ID: "busy", InputTokens: make([]sim.TokenID, 43)}
	for i := range busy.InputTokens {
		busy.InputTokens[i] = sim.TokenID(100 + i)
	}
	if !s.AllocateKVBlocks(busy, 0, 43, nil) {
		t.Fatal("busy allocation")
	}
	q := &sim.Request{ID: "waiting", InputTokens: input}
	for i := 0; i < 3; i++ {
		if s.AllocateKVBlocks(q, 0, 9, nil) {
			t.Fatal("admitted beyond active HBM capacity")
		}
	}
	for _, pool := range []string{"dram", "cxl"} {
		if h.f.Snapshot()[pool]["lookup_references"] != 1 {
			t.Fatal("lookup was delayed until allocation or duplicated on retry", h.f.Snapshot())
		}
	}
	for _, object := range h.f.hostChunks.objects {
		if object.evictable() {
			t.Fatal("queued lookup source was evictable")
		}
	}
	s.ReleaseKVBlocks(busy)
	seedPeer(t, s, "competing-store", []sim.TokenID{201, 202, 203, 204, 205})
	if h.f.Pending() != 0 || len(h.f.hostChunks.objects) != 2 {
		t.Fatal("new STORE evicted an already-owned lookup source")
	}
	s.ClearDeferred(q.ID)
	s.ClearDeferred(q.ID)
	for _, pool := range []string{"dram", "cxl"} {
		if h.f.Snapshot()[pool]["lookup_references"] != 0 {
			t.Fatal("cancel did not close its lease")
		}
	}
	if s.ensureHostLookup(q.ID, input) != nil {
		t.Fatal("late lookup resurrected ended demand")
	}
	assertPeerConservation(t, s)
}

func TestHostLookupMissIsFrozenAcrossPublication(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5}
	seedPeer(t, s, "writer", input)
	lookup := s.ensureHostLookup("early", input)
	if len(lookup.prefix) != 0 {
		t.Fatal("unpublished object hit")
	}
	drainHost(t, h, s)
	if s.ensureHostLookup("early", input) != lookup || len(lookup.prefix) != 0 {
		t.Fatal("retry refreshed native miss")
	}
	if e, _ := s.findForRestore("early", s.hashes(input)[0], input); e != nil {
		t.Fatal("restore bypassed frozen lookup")
	}
	if h.f.Snapshot()["dram"]["lookup_references"] != 0 {
		t.Fatal("miss acquired later ownership")
	}
	s.endHostLookup("early")
}

func TestHostLookupCannotSkipHostHoleUsingHBMCoverage(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	old := s.GetCachedBlocks(input)
	// Replace the first host object while retaining its HBM copy. This is a
	// legal alternate placement policy, and keeps the second host object.
	h.f.hostChunks.policy = hostPlacementFunc(func(v HostPlacementView) HostPlacementPlan {
		for _, c := range v.Candidates {
			if c.Pool == "dram" {
				return HostPlacementPlan{Destinations: []string{"dram"}, Victims: []string{c.Key}}
			}
		}
		return HostPlacementPlan{}
	})
	seedPeer(t, s, "replacement", []sim.TokenID{21, 22, 23, 24, 25})
	drainHost(t, h, s)
	for _, id := range old[2:] {
		s.invalidate(s.Blocks[id])
	}
	q := &sim.Request{ID: "hole", InputTokens: input}
	cached := s.GetRequestCachedPrefix(q)
	if len(cached.Blocks) != 2 {
		t.Fatal("fixture did not retain HBM-covered first host chunk")
	}
	if !s.AllocateKVBlocks(q, cached.Tokens, 9, cached.Blocks) || h.f.Pending() != 0 {
		t.Fatal("loaded a suffix beyond a host prefix hole")
	}
	if len(h.f.hostChunks.lookups[q.ID].prefix) != 0 {
		t.Fatal("host lookup incorrectly began after the HBM prefix")
	}
	metric := s.cacheMetrics.requests[q.ID]
	if metric.DRAMHitTokens != 0 || metric.OtherPoolHitTokens != 0 {
		t.Fatal("reported inaccessible suffix as a host cache hit")
	}
	s.ReleaseKVBlocks(q)
}

func TestHostLookupENDCannotReleaseAnotherReaderOrInFlightDMA(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, id := range s.GetCachedBlocks(input) {
		s.invalidate(s.Blocks[id])
	}
	s.ensureHostLookup("other", input)
	a := &sim.Request{ID: "reader", InputTokens: input}
	if s.AllocateKVBlocks(a, 0, 5, nil) {
		t.Fatal("LOAD not deferred")
	}
	state := h.f.Snapshot()["dram"]
	if state["lookup_references"] != 1 || state["transfer_reservations"] != 1 {
		t.Fatal("logical-to-physical handoff incorrect", state)
	}
	s.ClearDeferred(a.ID)
	s.ClearDeferred(a.ID)
	state = h.f.Snapshot()["dram"]
	if state["lookup_references"] != 1 || state["transfer_reservations"] != 1 {
		t.Fatal("cancel stole another reader or DMA pin", state)
	}
	s.endHostLookup("other")
	s.endHostLookup("other")
	if h.f.Snapshot()["dram"]["transfer_reservations"] != 1 {
		t.Fatal("END released active DMA")
	}
	drainHost(t, h, s)
	state = h.f.Snapshot()["dram"]
	if state["lookup_references"] != 0 || state["transfer_reservations"] != 0 || state["read_pins"] != 0 || s.UsedBlocks() != 0 {
		t.Fatal("ownership did not drain", state)
	}
}

func TestHostLookupLocalHitReleasesOnlyUnusedLease(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	q := &sim.Request{ID: "local", InputTokens: input}
	s.ensureHostLookup(q.ID, input)
	if h.f.Snapshot()["dram"]["lookup_references"] != 1 {
		t.Fatal("lookup reservation missing")
	}
	prefix := s.GetRequestCachedPrefix(q)
	if !s.AllocateKVBlocks(q, prefix.Tokens, 5, prefix.Blocks) {
		t.Fatal("local hit failed")
	}
	if h.f.Snapshot()["dram"]["lookup_references"] != 0 || h.f.hostChunks.lookups[q.ID] == nil {
		t.Fatal("admission must release unused locks but preserve session")
	}
	q.State = sim.StateRunning
	s.ReleaseKVBlocks(q) // preemption: no new external lookup on re-admission
	if h.f.hostChunks.lookups[q.ID] == nil || !h.f.hostChunks.lookups[q.ID].admitted {
		t.Fatal("preemption discarded consumed lookup state")
	}
	s.ClearDeferred(q.ID)
	if h.f.hostChunks.lookups[q.ID] != nil {
		t.Fatal("terminal waiting cleanup did not close session")
	}
}
