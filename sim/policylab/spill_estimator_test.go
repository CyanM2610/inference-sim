package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func spillEstimateFixture(t *testing.T) (*restoreEstimator, sim.DecisionView) {
	t.Helper()
	c := budgetLabConfig()
	c.DecisionPolicy.RequestSpill, c.DecisionPolicy.PrefillPreemption = true, true
	c.DecisionEstimates.Spill = true
	m, err := newRestoreEstimator(c)
	if err != nil {
		t.Fatal(err)
	}
	v := sim.DecisionView{BlockTokens: 16, MaxSequences: 2, MaxBatchTokens: 32, PrefillChunk: 32,
		Capabilities: sim.DecisionCapabilities{RequestSpill: true},
		Running:      []sim.DecisionRequest{{ID: "a", InputTokens: 64, ComputedTokens: 32, PrefillPreemptible: true}},
		KV: sim.DecisionKVState{PrefixStateKnown: true, TransfersKnown: true, Pools: []sim.DecisionPool{{ID: "dram"}},
			Requests: []sim.DecisionKVRequest{{ID: "a", SpillTargets: []sim.DecisionSpillTarget{{Pool: "dram", CompleteBlocks: 2, MissingBlocks: 2, AvailableBlocks: 8}}}}}}
	return m, v
}

func TestSpillEstimatorPricesNewStoreAndPublicBacklogWithoutFutureState(t *testing.T) {
	m, v := spillEstimateFixture(t)
	before := v
	e, err := m.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := sim.ValidateDecisionEstimates(v, e); err != nil {
		t.Fatal(err)
	}
	row := e.Spills[0]
	// 2 x 2,097,152 bytes / 25,000 => 168us + 3us latency;
	// pre/adoption 25us + grouped submission 7us => 203us.
	if row.StoreUS != 203 || row.CopyUS != 171 || row.HostUS != 32 || row.BacklogUS != 0 || !row.CapacityPossible {
		t.Fatal("incorrect grouped spill service", row)
	}
	if !reflect.DeepEqual(v, before) {
		t.Fatal("estimator changed its input")
	}
	v.KV.PendingTransfers = []sim.DecisionTransfer{{ID: 7, Request: "other", Source: "dram", Destination: "hbm", Bytes: m.phase.BlockBytes}}
	e, err = m.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	if e.Spills[0].BacklogUS != 93 || e.Spills[0].StoreUS != 296 {
		t.Fatal("shared link backlog/host submission disappeared", e.Spills[0])
	}
	v.KV.Requests[0].SpillTargets[0].MissingBlocks = 0
	v.KV.Requests[0].SpillTargets[0].ReadyBlocks = 2
	e, err = m.Estimate(v)
	if err != nil || e.Spills[0].StoreUS != 0 {
		t.Fatal("ready copy paid for another STORE", err, e.Spills)
	}
}

func TestSpillEstimatorReportsUnknownInventoryAndRejectsInvalidPartition(t *testing.T) {
	m, v := spillEstimateFixture(t)
	v.KV.Requests[0].SpillTargets = nil
	e, err := m.Estimate(v)
	if err != nil || e.Spills[0].Unavailable != "missing_spill_inventory" {
		t.Fatal("unknown became zero-cost available", e, err)
	}
	if err := sim.ValidateDecisionEstimates(v, e); err != nil {
		t.Fatal(err)
	}
	v.KV.Requests[0].SpillTargets = []sim.DecisionSpillTarget{{Pool: "dram", CompleteBlocks: 2, ReadyBlocks: 3}}
	if _, err = m.Estimate(v); err == nil {
		t.Fatal("invalid physical block partition accepted")
	}
}
