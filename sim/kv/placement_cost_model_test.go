package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type benefitPhaseFixture struct{}

func (benefitPhaseFixture) PredictEngineStep(w []sim.BatchWork) EngineStepTiming {
	if len(w) == 0 {
		return EngineStepTiming{PreForwardUS: 1, PostForwardUS: 2, PollUS: 3, TailUS: 5}
	}
	return EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 40, OutputReadyUS: 30, PollUS: 3, TailUS: 5}
}
func (benefitPhaseFixture) HostSubmitUS(_ string, blocks int64) int64 { return 11 + blocks }
func (benefitPhaseFixture) PredictLoadPlanning(_, blocks int64) (int64, error) {
	return 7 + blocks, nil
}

func costFixture(t *testing.T) *PlacementPhaseCosts {
	t.Helper()
	p, err := newPlacementPhaseCosts(benefitPhaseFixture{}, 4, 10, 100, []PeerResource{{BytesPerUS: 10, LatencyUS: 2}}, []PeerResource{{BytesPerUS: 10, LatencyUS: 2}})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestPlacementComputeUsesEngineBoundariesAndChunkGPUCarry(t *testing.T) {
	p := costFixture(t)
	one := p.compute(100, 4)
	// decision 10 + max(post10+poll3, output30) + tail5 =45.
	if one.TotalUS != 45 || one.ComputeSteps != 1 || one.ComputedTokens != 4 {
		t.Fatal("added concurrent boundaries as serial costs", one)
	}
	two := p.compute(100, 8)
	if two.TotalUS != 90 || two.ComputeSteps != 2 {
		t.Fatal("bad chunk phase composition", two)
	}
}

func TestPlacementVectorCopyCostsAndDelayedAdoption(t *testing.T) {
	p := costFixture(t)
	if p.storeService(4) >= 4*p.storeService(1) {
		t.Fatal("charged vector setup once per block")
	}
	one := p.load(1, 0, 0)
	// decision10 + plan8 + idle post2 + submit12 ->32; physical end44;
	// first poll35 misses completion, next poll55 + tail5 ->60.
	if one.ReadyUS != 60 || one.AdoptionWaitUS != 16 || one.SubmitUS != 12 || one.PlanningUS != 8 {
		t.Fatal("premature READY or lost overhead", one)
	}
	queued := p.load(1, 100, 0)
	if queued.ReadyUS != 120 || queued.QueueUS != 68 {
		t.Fatal("lost submitted queue delay", queued)
	}
	gpu := p.load(1, 0, 100)
	if gpu != queued {
		t.Fatal("main-stream dependency omitted", gpu, queued)
	}
}

func TestPlacementLoadPredictionMatchesExecutedPollAndAdoption(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 1, TailUS: 2})
	path := []PeerResource{h.f.resources["cxl"]}
	p, err := newPlacementPhaseCosts(h.f.phases.model, 4, 7, h.f.bytes, path, path)
	if err != nil {
		t.Fatal(err)
	}
	prediction := p.load(2, 0, 0)
	r := &sim.Request{ID: "conditional-load", InputTokens: tokens}
	s.clock = 0
	if s.AllocateKVBlocks(r, 0, 2, nil) {
		t.Fatal("fixture skipped real deferred load")
	}
	now := int64(0)
	for h.f.Pending() > 0 {
		end := int64(-1)
		ok, err := s.BeginBatch(now+7, nil, func(at int64) { end = at })
		if !ok || err != nil {
			t.Fatal(ok, err)
		}
		for end < 0 {
			phaseNext(t, h)
		}
		now = end
	}
	if now != prediction.ReadyUS || len(s.GetCachedBlocks(tokens)) != 2 {
		t.Fatal("conditional estimate disagrees with executed engine", now, prediction)
	}
	s.ClearDeferred(r.ID)
	assertPeerConservation(t, s)
}
