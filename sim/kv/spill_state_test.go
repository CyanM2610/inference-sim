package kv

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestSpillStateUsesCompletedPublishedPrefixWithoutMutatingResources(t *testing.T) {
	h, s, _ := newPeerHarness(t, false)
	r := capacityRequest(t, s, "victim", 8, 7, 10)
	s.MirrorToCPU([]*sim.Request{r})
	ids := s.RequestMap[r.ID]
	s.storeCopy(s.Blocks[ids[0]], "cxl", r.ID, true)
	h.next(t)
	s.storeCopy(s.Blocks[ids[1]], "cxl", r.ID, true)
	unrelated, _ := h.f.reserve("cxl", "other", []sim.TokenID{98, 99}, 0)
	unrelated.ready, unrelated.readers = true, 1
	before, events, pending := h.f.Snapshot(), len(h.records), h.f.Pending()
	queries := []sim.DecisionKVQuery{{ID: r.ID, Input: r.InputTokens, SpillComputedTokens: 7}}
	view := s.DecisionState(queries)
	var target sim.DecisionSpillTarget
	for _, row := range view.Requests[0].SpillTargets {
		if row.Pool == "cxl" {
			target = row
		}
	}
	want := sim.DecisionSpillTarget{Pool: "cxl", CompleteBlocks: 3, ReadyBlocks: 1, PendingBlocks: 1, MissingBlocks: 1, AvailableBlocks: 1, TailTokens: 1}
	if target != want {
		t.Fatal("wrong detached inventory", target)
	}
	if !reflect.DeepEqual(before, h.f.Snapshot()) || events != len(h.records) || pending != h.f.Pending() {
		t.Fatal("snapshot changed resources")
	}
	view.Requests[0].SpillTargets[1].ReadyBlocks = 999
	if next := s.DecisionState(queries); next.Requests[0].SpillTargets[1] != want {
		t.Fatal("snapshot aliases internal state")
	}
	unrelated.readers = 0
	if next := s.DecisionState(queries); next.Requests[0].SpillTargets[1].AvailableBlocks != 2 {
		t.Fatal("evictable capacity not reflected")
	}
	s.ReleaseKVBlocks(r)
	for len(h.events) > 0 {
		h.next(t)
	}
	assertPeerConservation(t, s)
}

func TestSpillStateDoesNotExposeUnadoptedPhysicalCompletion(t *testing.T) {
	h, s := reuseFixtureCapacity(t, 3)
	r := capacityRequest(t, s, "victim", 6, 4, 1)
	s.MirrorToCPU([]*sim.Request{r})
	q := []sim.DecisionKVQuery{{ID: r.ID, Input: r.InputTokens, SpillComputedTokens: 4}}
	before := s.DecisionState(q)
	for _, job := range h.f.phases.jobs {
		job.physicalDone = true
	}
	after := s.DecisionState(q)
	if !reflect.DeepEqual(before, after) || after.Requests[0].SpillTargets[0].PendingBlocks != 2 {
		t.Fatal("physical completion leaked through snapshot")
	}
	for _, job := range h.f.phases.jobs {
		job.physicalDone = false
	}
	s.ReleaseKVBlocks(r)
	finishRestoreStep(t, h, s)
}
