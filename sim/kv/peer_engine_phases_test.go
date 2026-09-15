package kv

import (
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type fixedEnginePhases struct {
	timing EngineStepTiming
	submit int64
}

func (m fixedEnginePhases) PredictEngineStep([]sim.BatchWork) EngineStepTiming { return m.timing }
func (m fixedEnginePhases) HostSubmitUS(string, int64) int64                   { return m.submit }

func phaseFixture(t *testing.T, timing EngineStepTiming) (*peerHarness, *PeerCache, []sim.TokenID) {
	t.Helper()
	h, s, tokens := concurrentRestoreFixture(t, 6)
	h.f.stores = []*PeerCache{s} // the other cache only populated the initial CPU fixture
	if err := s.ConfigureEnginePhases(fixedEnginePhases{timing, 3}, nil); err != nil {
		t.Fatal(err)
	}
	return h, s, tokens[0]
}
func phaseNext(t *testing.T, h *peerHarness) int64 {
	t.Helper()
	if len(h.events) == 0 {
		t.Fatal("no phase event")
	}
	sort.SliceStable(h.events, func(i, j int) bool {
		if h.events[i].Timestamp() != h.events[j].Timestamp() {
			return h.events[i].Timestamp() < h.events[j].Timestamp()
		}
		return h.events[i].Priority() < h.events[j].Priority()
	})
	e := h.events[0]
	h.events = h.events[1:]
	e.Execute(nil)
	return e.Timestamp()
}

func TestEnginePhasesMissedPollWaitsForLaterAdoption(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2})
	s.clock = 100
	r := &sim.Request{ID: "load", InputTokens: tokens}
	if s.AllocateKVBlocks(r, 0, 2, nil) || len(h.f.queue) != 0 || h.f.active != 0 {
		t.Fatal("reservation started DMA before host submission")
	}
	var end int64
	work := []sim.BatchWork{{Request: &sim.Request{ID: "compute"}, NewTokens: 1}}
	if ok, err := s.BeginBatch(100, work, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	if end != 132 || len(s.GetCachedBlocks(tokens)) != 0 {
		t.Fatal("poll published a future completion", end)
	}
	// At 130 the gate releases. The 200-byte CXL copy takes 43 us.
	for h.f.active > 0 {
		phaseNext(t, h)
	}
	if len(s.GetCachedBlocks(tokens)) != 0 || s.FreeBlockCnt != 4 {
		t.Fatal("physical completion released unpublished target")
	}
	end = 0
	if ok, err := s.BeginBatch(180, nil, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	if end != 193 || len(s.GetCachedBlocks(tokens)) != 2 || h.f.Pending() != 0 {
		t.Fatal("later worker poll did not adopt complete prefix", end)
	}
	if !s.AllocateKVBlocks(r, 4, 5, s.GetCachedBlocks(tokens)) {
		t.Fatal("adopted prefix unusable")
	}
	s.ReleaseKVBlocks(r)
	assertPeerConservation(t, s)
}

func TestEnginePhasesCompletionAtPollIsAdopted(t *testing.T) {
	// Submit at 13; 43 us of service ends at 56, exactly the worker poll.
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 43, TailUS: 2})
	s.clock = 0
	r := &sim.Request{ID: "load", InputTokens: tokens}
	s.AllocateKVBlocks(r, 0, 2, nil)
	var end int64
	s.BeginBatch(0, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	if end != 58 || len(s.GetCachedBlocks(tokens)) != 2 {
		t.Fatal("co-timed completion missed poll", end)
	}
	s.ClearDeferred(r.ID)
	assertPeerConservation(t, s)
}

func TestEnginePhasesFlushWaitsForSpecificPhysicalStores(t *testing.T) {
	h, s, _ := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 20, OutputReadyUS: 20, PollUS: 1, TailUS: 2})
	// A real reclaim dependency: the source remains pinned by its done callback.
	called := false
	h.f.submit(0, &peerJob{record: PeerRecord{Instance: s.id, Request: "reclaim", Reason: "store", Destination: "cxl"},
		path: []string{"cxl"}, schedule: s.schedule, blockedRequests: func() []string { return []string{"blocked"} }, done: func(int64) { called = true }})
	var end int64
	s.BeginBatch(0, []sim.BatchWork{{Request: &sim.Request{ID: "compute"}, NewTokens: 1}}, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	// pre=2, host ends=5, physical ends=28; forward/output shifted by 26.
	if end != 48 || !called {
		t.Fatal("flush did not wait on real completion", end, called)
	}
	var waits, ready int
	for _, r := range h.records {
		if r.Name == "engine_flush_wait" {
			waits++
		}
		if r.Name == "engine_flush_ready" {
			ready++
		}
	}
	if waits != 1 || ready != 1 {
		t.Fatal("missing flush boundary")
	}
}

func TestEnginePhasesCancellationDuringDMAReleasesAfterAdoption(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 50, TailUS: 2})
	r := &sim.Request{ID: "cancelled", InputTokens: tokens}
	s.clock = 0
	s.AllocateKVBlocks(r, 0, 2, nil)
	var end int64
	s.BeginBatch(0, nil, func(at int64) { end = at })
	for h.f.active == 0 {
		phaseNext(t, h)
	}
	s.ClearDeferred(r.ID)
	if s.FreeBlockCnt != 4 {
		t.Fatal("cancel freed DMA targets")
	}
	for end == 0 {
		phaseNext(t, h)
	}
	if s.FreeBlockCnt != 6 || h.f.Pending() != 0 || len(s.holds) != 0 {
		t.Fatal("late adoption leaked cancelled load")
	}
	assertPeerConservation(t, s)
}
