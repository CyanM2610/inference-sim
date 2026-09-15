package kv

import (
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestDirectionalTransferOrderPreservesHostOrderAcrossGPUFence(t *testing.T) {
	for _, load := range []bool{false, true} {
		h, s, _ := twoSubmissionFixture(t, load, submissionFunc(reverseSubmission), nil, nil, true)
		a, b := h.f.phases.staged[0].record.Transaction, h.f.phases.staged[1].record.Transaction
		h.f.phases.lastGPUReady = 200
		submissionStep(t, h, s, 0)
		for len(h.events) > 0 {
			phaseNext(t, h)
		}
		var host, wire []int64
		for _, e := range h.records {
			if e.Name == "transfer_submit_begin" {
				host = append(host, e.Transaction)
			}
			if e.Name == "transfer_start" {
				wire = append(wire, e.Transaction)
			}
		}
		if !reflect.DeepEqual(host, []int64{b, a}) || !reflect.DeepEqual(wire, host) {
			t.Fatal("common GPU fence lost submitted order", load, host, wire)
		}
		submissionStep(t, h, s, 400)
		if h.f.Pending() != 0 {
			t.Fatal("dependency chain leaked")
		}
	}
}

func unsubmittedOrderedLoads(t *testing.T, n int) (*peerHarness, *PeerCache, []*sim.Request) {
	t.Helper()
	h, source, s := newPeerHarness(t, false)
	h.f.pools["cxl"].config.CapacityBlocks = int64(2*n + 2)
	source.KVCacheState = NewKVCacheState(int64(3*n+3), 2)
	s.KVCacheState = NewKVCacheState(int64(3*n+3), 2)
	var requests []*sim.Request
	for i := 0; i < n; i++ {
		tokens := []sim.TokenID{sim.TokenID(10*i + 1), sim.TokenID(10*i + 2), sim.TokenID(10*i + 3), sim.TokenID(10*i + 4), sim.TokenID(10*i + 5)}
		r := seedPeer(t, source, fmt.Sprint(i), tokens)
		for _, id := range source.GetCachedBlocks(tokens) {
			if !source.store(source.Blocks[id], "cxl", r.ID) {
				t.Fatal("seed store")
			}
		}
		for len(h.events) > 0 {
			h.next(t)
		}
		requests = append(requests, &sim.Request{ID: fmt.Sprint(i), InputTokens: tokens})
	}
	if err := h.f.ConfigureMechanisms(PeerMechanisms{GroupTransfers: true, ConcurrentRestores: true, BackgroundStorePool: "cxl", RestoreWindow: 16}); err != nil {
		t.Fatal(err)
	}
	h.f.stores = []*PeerCache{s}
	if err := s.ConfigureEnginePhases(fixedEnginePhases{EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 1, TailUS: 2}, 3}, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableDirectionalTransferOrder(); err != nil {
		t.Fatal(err)
	}
	h.records = nil
	return h, s, requests
}

func phaseThrough(t *testing.T, h *peerHarness, until int64) {
	t.Helper()
	for len(h.events) > 0 {
		sort.SliceStable(h.events, func(i, j int) bool { return h.events[i].Timestamp() < h.events[j].Timestamp() })
		if h.events[0].Timestamp() > until {
			return
		}
		phaseNext(t, h)
	}
}

func TestDirectionalTransferOrderWaitsForLaterReadyPredecessorAcrossSubmissions(t *testing.T) {
	h, s, requests := unsubmittedOrderedLoads(t, 4)
	var ids []int64
	for i, request := range requests[:3] {
		now := int64(i * 10)
		phaseThrough(t, h, now)
		s.clock = now
		if s.AllocateKVBlocks(request, 0, 2, nil) {
			t.Fatal("expected staged restore")
		}
		ids = append(ids, h.f.phases.staged[0].record.Transaction)
		gpuReady := int64(0)
		if i == 0 {
			gpuReady = 200
		}
		h.f.phases.submit(now, gpuReady, true)
	}
	phaseThrough(t, h, 30)
	if h.f.active != 0 {
		t.Fatal("later-ready predecessor was bypassed")
	}
	for i := 0; i < 3; i++ {
		if predicted := h.f.EstimateQueueUS(30, []string{"cxl"}); predicted != 299 {
			t.Fatal("queue estimate omitted submitted GPU/dependency waits", predicted)
		}
	}
	if h.f.active != 0 || len(h.f.queue) != 3 {
		t.Fatal("queue projection advanced actual work")
	}
	before := s.FreeBlockCnt
	s.ClearDeferred(requests[1].ID)
	if s.FreeBlockCnt != before {
		t.Fatal("cancelled chained DMA released its destination early")
	}
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	var starts []int64
	for _, e := range h.records {
		if e.Name == "transfer_start" {
			starts = append(starts, e.Time)
		}
	}
	if !reflect.DeepEqual(starts, []int64{200, 243, 286}) {
		t.Fatal("predecessor chain not driven by physical completion", starts)
	}
	if h.f.Pending() != 3 || h.f.Snapshot()["cxl"]["read_pins"] != 6 {
		t.Fatal("physical completion bypassed adoption")
	}
	for _, state := range h.f.phases.jobs {
		if state.predecessor != nil {
			t.Fatal("completed chain retained backward references")
		}
	}
	submissionStep(t, h, s, 400)
	if len(s.holds[requests[1].ID]) != 0 || len(s.cancelledRestores) != 0 {
		t.Fatal("cancelled load recreated request ownership")
	}
	for _, id := range s.GetCachedBlocks(requests[1].InputTokens) {
		if s.Blocks[id].RefCount != 0 {
			t.Fatal("cancelled DMA result was not an unowned cache entry")
		}
	}
	for _, request := range requests {
		s.ClearDeferred(request.ID)
		s.ReleaseKVBlocks(request)
	}
	assertPeerConservation(t, s)
	if h.f.Pending() != 0 || h.f.Snapshot()["cxl"]["read_pins"] != 0 {
		t.Fatal("adoption leaked chain leases")
	}
	// A new transfer may refer to a tail already removed from jobs; it must
	// proceed without requiring that predecessor to be adopted a second time.
	s.clock = 500
	if s.AllocateKVBlocks(requests[3], 0, 2, nil) {
		t.Fatal("previously unused prefix unexpectedly cached")
	}
	h.f.phases.submit(500, 0, true)
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	submissionStep(t, h, s, 600)
	if h.f.Pending() != 0 {
		t.Fatal("adopted predecessor blocked a later phase")
	}
	s.ClearDeferred(requests[3].ID)
	assertPeerConservation(t, s)
}

func TestDirectionalTransferOrderKeepsPredecessorAcrossEngineSteps(t *testing.T) {
	h, s, requests := unsubmittedOrderedLoads(t, 2)
	s.clock = 0
	if s.AllocateKVBlocks(requests[0], 0, 2, nil) {
		t.Fatal("first restore missing")
	}
	h.f.phases.lastGPUReady = 200
	firstEnd := submissionStep(t, h, s, 0)
	if firstEnd >= 200 {
		t.Fatal("fixture did not leave a pending prior-step transfer")
	}
	s.clock = 30
	if s.AllocateKVBlocks(requests[1], 0, 2, nil) {
		t.Fatal("second restore missing")
	}
	submissionStep(t, h, s, 30)
	var links []PeerRecord
	for _, e := range h.records {
		if e.Name == "transfer_submission_dependency" {
			links = append(links, e)
		}
	}
	if len(links) != 1 || links[0].Time != 43 {
		t.Fatal("new engine step reset the predecessor", links)
	}
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	submissionStep(t, h, s, 400)
	for _, request := range requests {
		s.ClearDeferred(request.ID)
	}
	assertPeerConservation(t, s)
}

func TestDirectionalTransferOrderDoesNotSerializeIndependentDirections(t *testing.T) {
	h, s, requests := unsubmittedOrderedLoads(t, 1)
	// Independent read/write paths to the same CPU pool, representing separate
	// directional resources. The STORE must not inherit the LOAD's GPU fence.
	for i := range s.access {
		if s.access[i].Pool == "cxl" {
			s.access[i].WritePath = []string{"dram"}
		}
	}
	writer := seedPeer(t, s, "writer", []sim.TokenID{100, 101})
	if s.AllocateKVBlocks(requests[0], 0, 2, nil) {
		t.Fatal("load missing")
	}
	h.f.phases.submit(0, 200, true)
	h.f.phases.submit(0, 0, false)
	phaseThrough(t, h, 30)
	storeStarted, loadStarted := false, false
	for _, e := range h.records {
		if e.Name == "transfer_start" {
			if e.Destination == "hbm" {
				loadStarted = true
			} else {
				storeStarted = true
			}
		}
	}
	if !storeStarted || loadStarted {
		t.Fatal("opposite directions inherited a global dependency")
	}
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	submissionStep(t, h, s, 300)
	s.ClearDeferred(requests[0].ID)
	s.ReleaseKVBlocks(writer)
	assertPeerConservation(t, s)
}

func TestDirectionalTransferOrderPreservesStoreSourceReuseFence(t *testing.T) {
	h, s := reuseFixture(t)
	if err := s.EnableDirectionalTransferOrder(); err != nil {
		t.Fatal(err)
	}
	if err := s.SetTransferSubmissionPolicy(submissionFunc(reverseSubmission), nil, nil); err != nil {
		t.Fatal(err)
	}
	seedPeer(t, s, "old-a", []sim.TokenID{1, 2})
	seedPeer(t, s, "old-b", []sim.TokenID{11, 12})
	fresh := &sim.Request{ID: "fresh", InputTokens: []sim.TokenID{21, 22, 23, 24}}
	if !s.AllocateKVBlocks(fresh, 0, 4, nil) {
		t.Fatal("source reuse could not allocate")
	}
	var end int64
	if ok, err := s.BeginBatch(0, []sim.BatchWork{{Request: fresh, NewTokens: 4}}, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	if end != 225 {
		t.Fatal("forward did not wait for both submitted STORE sources", end)
	}
	for _, id := range s.RequestMap[fresh.ID] {
		if s.Blocks[id].RefCount != 1 {
			t.Fatal("old STORE altered fresh ownership")
		}
	}
	s.ReleaseKVBlocks(fresh)
	assertPeerConservation(t, s)
	if h.f.Pending() != 0 || s.FreeBlockCnt != 2 {
		t.Fatal("STORE reuse chain leaked")
	}
}

func TestDirectionalTransferOrderInstallationIsBeforeExecution(t *testing.T) {
	_, raw, _ := newPeerHarness(t, false)
	if err := raw.EnableDirectionalTransferOrder(); err == nil {
		t.Fatal("missing phase backend accepted")
	}
	h, s, _ := twoSubmissionFixture(t, true, nil, nil, nil)
	if err := s.EnableDirectionalTransferOrder(); err == nil {
		t.Fatal("already staged work accepted")
	}
	submissionStep(t, h, s, 0)
	if err := s.EnableDirectionalTransferOrder(); err == nil {
		t.Fatal("late model switch accepted")
	}
	_, ready, _ := unsubmittedOrderedLoads(t, 1)
	if err := ready.EnableDirectionalTransferOrder(); err == nil {
		t.Fatal("duplicate enable accepted")
	}
}
