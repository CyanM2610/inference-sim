package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
	"testing"
)

func nativeHarness(t *testing.T) (*peerHarness, *PeerCache) {
	t.Helper()
	h := &peerHarness{}
	var err error
	h.f, err = NewPeerFabric([]PeerResource{{ID: "pcie", BytesPerUS: 1}}, []PeerPoolConfig{{ID: "dram", CapacityBlocks: 4}}, 917504, func(r PeerRecord) { h.records = append(h.records, r) })
	if err != nil {
		t.Fatal(err)
	}
	if err = h.f.ConfigureMechanisms(PeerMechanisms{ReserveAtDispatch: true, CancelQueuedPromotions: true}); err != nil {
		t.Fatal(err)
	}
	s, err := NewPeerCache("one", 4, 16, h.f, []PeerAccess{{Pool: "dram", ReadPath: []string{"pcie"}, WritePath: []string{"pcie"}}}, BuiltinPeerPolicy{Name: "lfu_store", MinFrequency: 1})
	if err != nil {
		t.Fatal(err)
	}
	s.BindEvents(func(e sim.Event) { h.events = append(h.events, e) }, func(int64) {})
	p := &kvruntime.Profile{QueueDepth: 4}
	for _, d := range []string{"d2h", "h2d"} {
		p.Transfers = append(p.Transfers, kvruntime.TransferPoint{NUMA: 0, Layout: "paged", HostMemory: "pinned", Blocks: 1, Direction: d, Cost: kvruntime.TransferCost{PlanNS: 1200, SetupNS: 800, SubmitNS: 400, DMANS: 2300, ObserveNS: 2600}})
	}
	if err = h.f.ConfigureNative(p, 0, nil); err != nil {
		t.Fatal(err)
	}
	return h, s
}
func TestNativePeerRetriesRestoreAndCorruption(t *testing.T) {
	h, s := nativeHarness(t)
	tokens := make([]sim.TokenID, 64)
	for i := range tokens {
		tokens[i] = sim.TokenID(i + 1)
	}
	seedPeer(t, s, "seed", tokens)
	for _, b := range s.Blocks {
		s.frequency[b.Hash] = 1
	}
	if s.makeSpace(2, nil, "need") {
		t.Fatal("store must be asynchronous")
	}
	for len(h.events) > 0 {
		h.next(t)
		s.makeSpace(2, nil, "need")
		if reclaimCount(h) != 2 {
			t.Fatal("retry added redundant victim")
		}
	}
	if err := h.f.NativeCheck(); err != nil {
		t.Fatal(err)
	}
	q := &sim.Request{ID: "restore", InputTokens: tokens}
	if s.AllocateKVBlocks(q, 0, 64, s.GetCachedBlocks(tokens)) {
		t.Fatal("restored before DMA")
	}
	for len(h.events) > 0 {
		s.SetClock(h.next(t))
	}
	// Restore window defaults to one, so repeat admissions until dependencies drain.
	for attempts := 0; attempts < 8; attempts++ {
		cached := s.GetCachedBlocks(tokens)
		if s.AllocateKVBlocks(q, int64(len(cached))*16, 64, cached) {
			break
		}
		if len(h.events) == 0 {
			t.Fatalf("restore made no progress cached=%v pending=%v snapshot=%v", s.GetCachedBlocks(tokens), s.waiting, s.PeerSnapshot())
		}
		for len(h.events) > 0 {
			s.SetClock(h.next(t))
		}
	}
	if len(s.RequestMap[q.ID]) == 0 {
		t.Fatal("request never admitted")
	}
	s.ReleaseKVBlocks(q)
	if err := h.f.NativeCheck(); err != nil {
		t.Fatal(err)
	}
	assertPeerConservation(t, s)
	for _, e := range h.records {
		if e.Name == "native_handoff" && (e.Value < 0 || e.Value >= 1000) {
			t.Fatal("per-fragment rounding leaked into handoff")
		}
	}
	for _, entry := range h.f.pools["dram"].entries {
		h.f.native.Runtime.Memory.Buffers[entry.nativeBuffer].Cells[0].Origin++
		break
	}
	if err := h.f.NativeCheck(); err == nil {
		t.Fatal("corruption went undetected")
	}
}
func TestNativeQueuedPromotionCancellationReleasesOnlyReservation(t *testing.T) {
	h, s := nativeHarness(t)
	tokens := make([]sim.TokenID, 64)
	for i := range tokens {
		tokens[i] = sim.TokenID(i + 1)
	}
	seedPeer(t, s, "seed", tokens)
	for _, b := range s.Blocks {
		s.frequency[b.Hash] = 1
	}
	cached := s.GetCachedBlocks(tokens)
	for _, id := range cached[:2] {
		if !s.store(s.Blocks[id], "dram", "need") {
			t.Fatal("could not seed promotion source")
		}
	}
	for len(h.events) > 0 {
		s.SetClock(h.next(t))
	}
	if got := s.Promote(tokens, 2); got != 2 {
		t.Fatalf("did not queue two promotions got=%d cached=%v pools=%v", got, s.GetCachedBlocks(tokens), h.f.Snapshot())
	}
	h.f.CancelQueuedPromotions(s.clock, s.id)
	for len(h.events) > 0 {
		s.SetClock(h.next(t))
	}
	if err := h.f.NativeCheck(); err != nil {
		t.Fatal(err)
	}
	assertPeerConservation(t, s)
	cancelled := 0
	for _, r := range h.records {
		if r.Name == "promotion_cancelled" {
			cancelled++
		}
	}
	if cancelled != 1 {
		t.Fatalf("cancelled %d; active transfer must finish", cancelled)
	}
}
