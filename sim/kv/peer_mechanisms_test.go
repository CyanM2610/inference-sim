package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestPeerCompletionAdoptionProtectsSource(t *testing.T) {
	h, a, _ := newPeerHarness(t, false)
	r := seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3})
	id := a.GetCachedBlocks(r.InputTokens)[0]
	if e := h.f.ConfigureMechanisms(PeerMechanisms{CompletionUS: 7, ControlWorkers: 1}); e != nil {
		t.Fatal(e)
	}
	a.store(a.Blocks[id], "cxl", "save")
	if at := h.next(t); at != 23 {
		t.Fatal(at)
	}
	if h.f.Snapshot()["cxl"]["ready"] != 0 || a.Blocks[id].RefCount != 1 || h.f.busy["cxl"] {
		t.Fatal("data completion must release link while retaining unpublished copy/source pin")
	}
	if at := h.next(t); at != 30 {
		t.Fatal(at)
	}
	if h.f.Snapshot()["cxl"]["ready"] != 1 || a.Blocks[id].RefCount != 0 || h.f.Pending() != 0 {
		t.Fatal("adoption did not release/publish")
	}
}
func TestPeerDirectorySharedWorkers(t *testing.T) {
	for _, workers := range []int{1, 2} {
		h, a, b := newPeerHarness(t, false)
		h.f.ConfigureMechanisms(PeerMechanisms{DirectoryUS: 10, ControlWorkers: workers})
		for i, s := range []*PeerCache{a, b} {
			r := &sim.Request{ID: []string{"q0", "q1"}[i], InputTokens: []sim.TokenID{1, 2, 3}}
			if s.AllocateKVBlocks(r, 0, 3, nil) || !s.IsDeferred(r.ID) {
				t.Fatal("directory bypassed")
			}
		}
		first, last := h.next(t), h.next(t)
		want := int64(20)
		if workers == 2 {
			want = 10
		}
		if first != 10 || last != want {
			t.Fatalf("workers=%d: %d,%d", workers, first, last)
		}
	}
}
func TestPeerDependencyEDFUnblocksStore(t *testing.T) {
	for _, policy := range []string{"read_first", "dependency_edf"} {
		h, a, _ := newPeerHarness(t, true)
		r := seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3})
		id := a.GetCachedBlocks(r.InputTokens)[0]
		h.f.ConfigureMechanisms(PeerMechanisms{TransferPolicy: policy})
		h.f.SetRequestDeadline("urgent", 100)
		h.f.SetRequestDeadline("normal", 200)
		schedule := func(e sim.Event) { h.events = append(h.events, e) }
		h.f.submit(0, &peerJob{record: PeerRecord{Request: "busy", Reason: "restore"}, path: []string{"pcie"}, schedule: schedule, done: func(int64) {}})
		h.f.submit(0, &peerJob{record: PeerRecord{Request: "normal", Reason: "restore"}, path: []string{"dram", "pcie"}, schedule: schedule, done: func(int64) {}})
		a.spaceWait["urgent"] = true
		a.store(a.Blocks[id], "cxl", "urgent")
		for len(h.events) > 0 {
			h.next(t)
		}
		var released int64
		for _, e := range h.records {
			if e.Name == "source_release" {
				released = e.Time
			}
		}
		want := int64(40)
		if policy == "dependency_edf" {
			want = 28
		}
		if released != want {
			t.Fatalf("%s released at %d, want %d", policy, released, want)
		}
	}
}
func TestPeerReservationTimingAndQueuedCancellation(t *testing.T) {
	for _, late := range []bool{false, true} {
		h, a, b := newPeerHarness(t, false)
		r := seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3})
		id := a.GetCachedBlocks(r.InputTokens)[0]
		hash := a.Blocks[id].Hash
		a.store(a.Blocks[id], "cxl", "save")
		h.next(t)
		h.f.ConfigureMechanisms(PeerMechanisms{ReserveAtDispatch: late})
		h.f.submit(23, &peerJob{record: PeerRecord{Request: "busy", Reason: "restore"}, path: []string{"cxl"}, schedule: func(e sim.Event) { h.events = append(h.events, e) }, done: func(int64) {}})
		e, access := b.find(hash)
		if !b.fetch(hash, "@promotion", e, access, map[int64]bool{}, true) {
			t.Fatal("fetch rejected")
		}
		want := int64(1)
		if late {
			want = 0
		}
		if b.PeerSnapshot()["active_or_pinned"] != want {
			t.Fatal("target reserved at wrong time")
		}
		if n := h.f.CancelQueuedPromotions(24, b.id); n != 1 {
			t.Fatal("queued promotion not cancelled")
		}
		if b.PeerSnapshot()["active_or_pinned"] != 0 || e.readers != 0 || len(b.restoring) != 0 {
			t.Fatal("cancellation leaked reservations")
		}
		h.next(t)
		if h.f.Pending() != 0 {
			t.Fatal("pending work")
		}
		assertPeerConservation(t, b)
	}
}
func TestPeerRunningPromotionDrains(t *testing.T) {
	h, a, b := newPeerHarness(t, false)
	r := seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3})
	id := a.GetCachedBlocks(r.InputTokens)[0]
	hash := a.Blocks[id].Hash
	a.store(a.Blocks[id], "cxl", "save")
	h.next(t)
	e, access := b.find(hash)
	b.fetch(hash, "@promotion", e, access, map[int64]bool{}, true)
	if n := h.f.CancelQueuedPromotions(24, b.id); n != 0 {
		t.Fatal("running DMA must not be cancelled")
	}
	if e.readers != 1 || b.PeerSnapshot()["active_or_pinned"] != 1 {
		t.Fatal("released running transfer protection")
	}
	h.next(t)
	assertPeerConservation(t, b)
}

func TestPeerQueueAwareReadSource(t *testing.T) {
	h, a, b := newPeerHarness(t, false)
	r := seedPeer(t, a, "a", []sim.TokenID{1, 2, 3})
	seedPeer(t, b, "b", r.InputTokens)
	aid := a.GetCachedBlocks(r.InputTokens)[0]
	bid := b.GetCachedBlocks(r.InputTokens)[0]
	hash := a.Blocks[aid].Hash
	a.store(a.Blocks[aid], "dram", "save-a")
	b.store(b.Blocks[bid], "cxl", "save-b")
	h.next(t)
	h.next(t)
	h.f.submit(23, &peerJob{record: PeerRecord{Reason: "restore"}, path: []string{"dram"}, schedule: func(e sim.Event) { h.events = append(h.events, e) }, done: func(int64) {}})
	_, before := a.find(hash)
	if before.Pool != "dram" {
		t.Fatal("isolated cost should choose DRAM")
	}
	h.f.ConfigureMechanisms(PeerMechanisms{QueueAware: true})
	_, after := a.find(hash)
	if after.Pool != "cxl" {
		t.Fatal("queued DRAM cost 24 us should exceed CXL 23 us")
	}
}
