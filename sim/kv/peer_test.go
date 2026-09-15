package kv

import (
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type peerHarness struct {
	events  []sim.Event
	records []PeerRecord
	f       *PeerFabric
}

func newPeerHarness(t *testing.T, shared bool) (*peerHarness, *PeerCache, *PeerCache) {
	t.Helper()
	h := &peerHarness{}
	var err error
	h.f, err = NewPeerFabric([]PeerResource{{ID: "dram", BytesPerUS: 10, LatencyUS: 2}, {ID: "cxl", BytesPerUS: 5, LatencyUS: 3}, {ID: "pcie", BytesPerUS: 20}}, []PeerPoolConfig{{ID: "dram", CapacityBlocks: 1}, {ID: "cxl", CapacityBlocks: 4}}, 100, func(r PeerRecord) { h.records = append(h.records, r) })
	if err != nil {
		t.Fatal(err)
	}
	a := []PeerAccess{{Pool: "dram", ReadPath: []string{"dram"}, WritePath: []string{"dram"}}, {Pool: "cxl", ReadPath: []string{"cxl"}, WritePath: []string{"cxl"}}}
	if shared {
		for i := range a {
			a[i].ReadPath = append(a[i].ReadPath, "pcie")
			a[i].WritePath = append(a[i].WritePath, "pcie")
		}
	}
	makeStore := func(id string) *PeerCache {
		s, e := NewPeerCache(id, 4, 2, h.f, a, BuiltinPeerPolicy{Name: "lfu_store", MinFrequency: 1})
		if e != nil {
			t.Fatal(e)
		}
		s.BindEvents(func(e sim.Event) { h.events = append(h.events, e) }, func(int64) {})
		return s
	}
	return h, makeStore("a"), makeStore("b")
}
func (h *peerHarness) next(t *testing.T) int64 {
	t.Helper()
	if len(h.events) == 0 {
		t.Fatal("missing event")
	}
	sort.SliceStable(h.events, func(i, j int) bool { return h.events[i].Timestamp() < h.events[j].Timestamp() })
	e := h.events[0]
	h.events = h.events[1:]
	e.Execute(nil)
	return e.Timestamp()
}
func seedPeer(t *testing.T, s *PeerCache, id string, toks []sim.TokenID) *sim.Request {
	t.Helper()
	r := &sim.Request{ID: id, InputTokens: toks}
	if !s.AllocateKVBlocks(r, 0, int64(len(toks)), nil) {
		t.Fatal("seed allocation")
	}
	r.ProgressIndex = int64(len(toks))
	s.MirrorToCPU([]*sim.Request{r})
	s.ReleaseKVBlocks(r)
	return r
}
func assertPeerConservation(t *testing.T, s *PeerCache) {
	t.Helper()
	if err := s.verifyBlockConservation(); err != nil {
		t.Fatal(err)
	}
	for _, b := range s.Blocks {
		if b.RefCount < 0 || b.InUse != (b.RefCount > 0) {
			t.Fatalf("bad refcount %+v", b)
		}
	}
}
func TestPeerStorePublishAndDedup(t *testing.T) {
	h, a, b := newPeerHarness(t, false)
	r := seedPeer(t, a, "a", []sim.TokenID{1, 2, 3})
	seedPeer(t, b, "b", r.InputTokens)
	id := a.GetCachedBlocks(r.InputTokens)[0]
	other := b.GetCachedBlocks(r.InputTokens)[0]
	hash := a.Blocks[id].Hash
	if !a.store(a.Blocks[id], "cxl", "writer1") || !b.store(b.Blocks[other], "cxl", "writer2") {
		t.Fatal("store failed")
	}
	if h.f.Snapshot()["cxl"]["reserved"] != 1 || h.f.Pending() != 1 {
		t.Fatal("duplicate physical reservation/transfer")
	}
	if a.Blocks[id].RefCount != 1 || b.Blocks[other].RefCount != 1 {
		t.Fatal("source not pinned")
	}
	if e, _ := b.find(hash); e != nil {
		t.Fatal("premature L2 hit")
	}
	if at := h.next(t); at != 23 {
		t.Fatalf("CXL 100/5+3: got %d", at)
	}
	if h.f.Snapshot()["cxl"]["ready"] != 1 || a.Blocks[id].RefCount != 0 || b.Blocks[other].RefCount != 0 {
		t.Fatal("completion did not publish/release")
	}
	assertPeerConservation(t, a)
	assertPeerConservation(t, b)
}
func TestPeerRestoreReservationAndReadPin(t *testing.T) {
	h, a, b := newPeerHarness(t, false)
	r := seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3})
	id := a.GetCachedBlocks(r.InputTokens)[0]
	a.store(a.Blocks[id], "dram", "save")
	h.next(t)
	q := &sim.Request{ID: "reader", InputTokens: r.InputTokens}
	if b.AllocateKVBlocks(q, 0, 3, nil) {
		t.Fatal("admitted before restore")
	}
	if len(b.GetCachedBlocks(q.InputTokens)) != 0 || h.f.Snapshot()["dram"]["read_pins"] != 1 {
		t.Fatal("reservation invisible/read source pinned invariant")
	}
	if e, _ := h.f.reserve("dram", "different", []sim.TokenID{8, 9}, 12); e != nil {
		t.Fatal("evicted read-pinned source")
	}
	if at := h.next(t); at != 24 {
		t.Fatalf("restore should finish at 24, got %d", at)
	}
	cached := b.GetCachedBlocks(q.InputTokens)
	if len(cached) != 1 || !b.AllocateKVBlocks(q, 2, 3, cached) {
		t.Fatal("restore not consumable")
	}
	b.ReleaseKVBlocks(q)
	if b.PeerSnapshot()["active_or_pinned"] != 0 {
		t.Fatal("restore leaked HBM")
	}
	assertPeerConservation(t, b)
}
func TestPeerPathsIndependentOrShared(t *testing.T) {
	for _, shared := range []bool{false, true} {
		h, a, b := newPeerHarness(t, shared)
		ra := seedPeer(t, a, "a", []sim.TokenID{1, 2, 3})
		rb := seedPeer(t, b, "b", []sim.TokenID{4, 5, 6})
		a.store(a.Blocks[a.GetCachedBlocks(ra.InputTokens)[0]], "dram", "a")
		b.store(b.Blocks[b.GetCachedBlocks(rb.InputTokens)[0]], "cxl", "b")
		first, last := h.next(t), h.next(t)
		want := int64(23)
		if shared {
			want = 35
		}
		if first != 12 || last != want {
			t.Fatalf("shared=%v got %d,%d", shared, first, last)
		}
	}
}
func TestPeerNoEarlyComputedHit(t *testing.T) {
	_, a, _ := newPeerHarness(t, false)
	r := &sim.Request{ID: "compute", InputTokens: []sim.TokenID{1, 2, 3}}
	a.AllocateKVBlocks(r, 0, 3, nil)
	if len(a.GetCachedBlocks(r.InputTokens)) != 0 {
		t.Fatal("allocation published uncomputed data")
	}
	r.ProgressIndex = 3
	a.MirrorToCPU([]*sim.Request{r})
	if len(a.GetCachedBlocks(r.InputTokens)) != 1 {
		t.Fatal("completion did not publish")
	}
}

func TestPeerStoreSourceClaimedDuringTransfer(t *testing.T) {
	h, a, _ := newPeerHarness(t, false)
	r := seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3})
	cached := a.GetCachedBlocks(r.InputTokens)
	b := a.Blocks[cached[0]]
	a.store(b, "cxl", "save")
	q := &sim.Request{ID: "reuse", InputTokens: r.InputTokens}
	if !a.AllocateKVBlocks(q, 2, 3, cached) {
		t.Fatal("source should remain reusable while pinned")
	}
	if b.RefCount != 2 {
		t.Fatalf("expected source and request references, got %d", b.RefCount)
	}
	h.next(t)
	if b.RefCount != 1 || len(a.GetCachedBlocks(q.InputTokens)) != 1 {
		t.Fatal("STORE completion reclaimed a live request's source")
	}
	a.ReleaseKVBlocks(q)
	assertPeerConservation(t, a)
}

func TestPeerSharedHBMIsOnePhysicalBlock(t *testing.T) {
	_, a, _ := newPeerHarness(t, false)
	r := seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3})
	cached := a.GetCachedBlocks(r.InputTokens)
	q1 := &sim.Request{ID: "q1", InputTokens: r.InputTokens}
	q2 := &sim.Request{ID: "q2", InputTokens: r.InputTokens}
	if !a.AllocateKVBlocks(q1, 2, 3, cached) || !a.AllocateKVBlocks(q2, 2, 3, cached) {
		t.Fatal("admission failed")
	}
	if a.Blocks[cached[0]].RefCount != 2 || a.TotalBlocks-a.FreeBlockCnt != 3 {
		t.Fatal("shared prefix counted twice physically")
	}
	a.ReleaseKVBlocks(q1)
	if a.Blocks[cached[0]].RefCount != 1 {
		t.Fatal("premature shared block free")
	}
	a.ReleaseKVBlocks(q2)
	assertPeerConservation(t, a)
}
