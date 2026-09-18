package kv

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestEnginePhaseObserverPreservesEventsAndSeparatesIdleSteps(t *testing.T) {
	run := func(observe bool) ([]PeerRecord, []EnginePhaseObservation) {
		h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2})
		var rows []EnginePhaseObservation
		if observe {
			if err := s.SetEnginePhaseObserver(func(p EnginePhaseObservation) { rows = append(rows, p) }); err != nil {
				t.Fatal(err)
			}
		}
		s.clock = 100
		r := &sim.Request{ID: "load", InputTokens: tokens}
		s.AllocateKVBlocks(r, 0, 2, nil)
		var done int64
		_, err := s.BeginBatch(100, []sim.BatchWork{{Request: &sim.Request{ID: "compute"}, NewTokens: 1}}, func(at int64) { done = at })
		if err != nil {
			t.Fatal(err)
		}
		for done == 0 {
			phaseNext(t, h)
		}
		if observe && (len(rows) != 1 || rows[0].GPUStartEstimateUS != 102 || rows[0].GPUReadyUS != 130 || rows[0].AdoptUS != 132) {
			t.Fatal(rows)
		}
		for i := 0; s.HasPendingEngineWork(); i++ {
			if i > 100 {
				t.Fatal("phase failed to drain")
			}
			finishRestoreStep(t, h, s)
		}
		return h.records, rows
	}
	a, _ := run(false)
	b, rows := run(true)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("observer changed execution trace")
	}
	if len(rows) < 2 {
		t.Fatal("fixture lacks idle control steps")
	}
	for _, row := range rows[1:] {
		if row.HasCompute {
			t.Fatal("idle step claims GPU work")
		}
	}
}
