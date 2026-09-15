package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

func TestHostCompletionBarrierKeepsReadyLoadPrivateUntilServiceEnd(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 43, TailUS: 2})
	var entered, end int64
	if err := s.BindHostCompletionBarrier(func(at int64, w []sim.BatchWork, publish func(int64)) {
		entered = at
		s.fabric.phases.schedule(at+30, publish)
	}); err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "load", InputTokens: tokens}
	s.AllocateKVBlocks(r, 0, 2, nil)
	s.BeginBatch(0, nil, func(at int64) { end = at })
	for entered == 0 {
		phaseNext(t, h)
	}
	if entered != 58 || end != 0 || len(s.GetCachedBlocks(tokens)) != 0 || !s.fabric.phases.active || s.FreeBlockCnt != 4 {
		t.Fatal("load published or target released before host completion", entered, end, s.FreeBlockCnt)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	if end != 88 || len(s.GetCachedBlocks(tokens)) != 2 {
		t.Fatal("barrier failed to publish once at service end", end)
	}
	s.ClearDeferred(r.ID)
	assertPeerConservation(t, s)
}

func TestHostCompletionBarrierCannotAdoptDMAMissedByOriginalPoll(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 1, TailUS: 2})
	if err := s.BindHostCompletionBarrier(func(at int64, w []sim.BatchWork, publish func(int64)) { s.fabric.phases.schedule(at+70, publish) }); err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "late", InputTokens: tokens}
	s.AllocateKVBlocks(r, 0, 2, nil)
	var end int64
	s.BeginBatch(0, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	if h.f.active != 0 || len(s.GetCachedBlocks(tokens)) != 0 {
		t.Fatal("service delay adopted a DMA absent from its poll snapshot")
	}
	start := end
	end = 0
	s.BeginBatch(start, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	if len(s.GetCachedBlocks(tokens)) != 2 {
		t.Fatal("later poll could not adopt completed DMA")
	}
	s.ClearDeferred(r.ID)
	assertPeerConservation(t, s)
}
