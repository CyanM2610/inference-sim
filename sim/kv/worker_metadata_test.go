package kv

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type metadataPhaseFixture struct {
	fixedEnginePhases
	observed [][]bool
	reject   bool
}

func (m *metadataPhaseFixture) WorkerMetadataEnabled() bool { return true }
func (m *metadataPhaseFixture) PredictEngineStepWithMetadata(_ []sim.BatchWork, fresh []bool) (EngineStepTiming, error) {
	if m.reject {
		return EngineStepTiming{}, fmt.Errorf("outside fixture coverage")
	}
	m.observed = append(m.observed, append([]bool(nil), fresh...))
	return m.timing, nil
}
func metadataFixture(t *testing.T) (*peerHarness, *PeerCache, *metadataPhaseFixture, []sim.TokenID) {
	t.Helper()
	h, s, tokens := concurrentRestoreFixture(t, 6)
	h.f.stores = []*PeerCache{s}
	m := &metadataPhaseFixture{fixedEnginePhases: fixedEnginePhases{EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 20, OutputReadyUS: 20, PollUS: 1, TailUS: 2}, 3}}
	if err := s.ConfigureEnginePhases(m, nil); err != nil {
		t.Fatal(err)
	}
	return h, s, m, tokens[0]
}
func finishMetadataStep(t *testing.T, h *peerHarness, s *PeerCache, at int64, work []sim.BatchWork) int64 {
	t.Helper()
	end := int64(0)
	if ok, err := s.BeginBatch(at, work, func(now int64) { end = now }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	return end
}

func TestWorkerMetadataTracksExecutionRatherThanPrefix(t *testing.T) {
	h, s, m, tokens := metadataFixture(t)
	a, b := &sim.Request{ID: "a", InputTokens: tokens}, &sim.Request{ID: "b", InputTokens: tokens}
	queries := []sim.DecisionKVQuery{{ID: a.ID, Input: tokens}, {ID: b.ID, Input: tokens}}
	view := s.DecisionState(queries)
	if !view.WorkerStateKnown || view.Requests[0].WorkerResident {
		t.Fatal("arrival became worker resident")
	}
	end := finishMetadataStep(t, h, s, 100, []sim.BatchWork{{Request: a, NewTokens: 16}})
	end = finishMetadataStep(t, h, s, end+1, []sim.BatchWork{{Request: a, PrefixTokens: 16, NewTokens: 16}, {Request: b, PrefixTokens: 32, NewTokens: 1}})
	if !reflect.DeepEqual(m.observed, [][]bool{{true}, {false, true}}) {
		t.Fatal(m.observed)
	}
	view = s.DecisionState(queries)
	if !view.Requests[0].WorkerResident || !view.Requests[1].WorkerResident {
		t.Fatal("executed worker state missing")
	}
	s.SetClock(end)
	s.ReleaseKVBlocks(a)
	view = s.DecisionState(queries)
	if view.Requests[0].WorkerResident || !view.Requests[1].WorkerResident {
		t.Fatal("release lost wrong worker identity")
	}
	finishMetadataStep(t, h, s, end+1, []sim.BatchWork{{Request: a, PrefixTokens: 32, NewTokens: 16}})
	if !m.observed[len(m.observed)-1][0] {
		t.Fatal("resumed nonzero prefix was priced as resident")
	}
	s.ReleaseKVBlocks(a)
	s.ReleaseKVBlocks(b)
	if len(s.fabric.phases.workerResidents) != 0 {
		t.Fatal("worker identities leaked after release")
	}
}

func TestWorkerMetadataLoadAndRejectedPredictionDoNotCreateWorker(t *testing.T) {
	h, s, m, tokens := metadataFixture(t)
	r := &sim.Request{ID: "load", InputTokens: tokens}
	s.SetClock(100)
	if s.AllocateKVBlocks(r, 0, 2, nil) {
		t.Fatal("fixture unexpectedly restored synchronously")
	}
	finishMetadataStep(t, h, s, 100, nil)
	if len(s.fabric.phases.workerResidents) != 0 {
		t.Fatal("load-only worker became resident")
	}
	m.reject = true
	if ok, err := s.BeginBatch(200, []sim.BatchWork{{Request: r, PrefixTokens: 32, NewTokens: 16}}, func(int64) {}); ok || err == nil {
		t.Fatal("unsupported prediction was accepted")
	}
	if s.fabric.phases.active || len(s.fabric.phases.workerResidents) != 0 {
		t.Fatal("rejected cost committed execution state")
	}
}
