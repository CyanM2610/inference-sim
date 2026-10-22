package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

type planningPhases struct {
	fixedEnginePhases
	cost   int64
	counts [][2]int64
}

func (m *planningPhases) PredictLoadPlanning(loads, blocks int64) (int64, error) {
	m.counts = append(m.counts, [2]int64{loads, blocks})
	return m.cost, nil
}

func TestLoadPlanningDelaysSubmissionOnceAndDoesNotPublishKV(t *testing.T) {
	h, s, r := restoreDecisionFixture(t)
	m := &planningPhases{fixedEnginePhases: fixedEnginePhases{EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 1, TailUS: 2}, 3}, cost: 40}
	s.fabric.phases.model = m
	if s.AllocateKVBlocks(r, 0, 4, nil) {
		t.Fatal("load admitted early")
	}
	finishRestoreStep(t, h, s)
	if len(m.counts) != 1 || m.counts[0] != [2]int64{1, 2} {
		t.Fatal("incorrect new-load work", m.counts)
	}
	var submit, planning int64
	for _, e := range h.records {
		if e.Name == "transfer_submit_begin" {
			submit = e.Time
		}
		if e.Name == "engine_load_planning" {
			planning += e.Duration
		}
	}
	if submit != 50 || planning != 40 || s.GetRequestCachedPrefix(r).Tokens != 0 {
		t.Fatal("planning did not precede submit, or published unreceived KV", submit, planning)
	}
	for h.f.active > 0 {
		phaseNext(t, h)
	}
	s.SetClock(1000)
	finishRestoreStep(t, h, s)
	if len(m.counts) != 1 || s.GetRequestCachedPrefix(r).Tokens != 3 {
		t.Fatal("poll charged new-load work or failed to adopt")
	}
	s.ClearDeferred(r.ID)
	assertPeerConservation(t, s)
}

func TestLoadPlanningCanOverlapPreviouslyLaunchedGPUWork(t *testing.T) {
	var ends []int64
	for _, cost := range []int64{0, 40} {
		h, s, r := restoreDecisionFixture(t)
		p := s.fabric.phases
		p.model = &planningPhases{fixedEnginePhases: fixedEnginePhases{EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2}, 3}, cost: cost}
		p.lastGPUReady = 1000
		s.AllocateKVBlocks(r, 0, 4, nil)
		var end int64
		ok, err := s.BeginBatch(0, []sim.BatchWork{{Request: &sim.Request{ID: "other"}, NewTokens: 1}}, func(at int64) { end = at })
		if !ok || err != nil {
			t.Fatal(ok, err)
		}
		for end == 0 {
			phaseNext(t, h)
		}
		ends = append(ends, end)
	}
	if ends[0] != 1032 || ends[1] != ends[0] {
		t.Fatal("host planning shifted an already busy GPU twice", ends)
	}
}
