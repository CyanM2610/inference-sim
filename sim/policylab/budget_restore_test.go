package policylab

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func budgetLabConfig() Config {
	c := restoreLabConfig(nil)
	c.DecisionEstimates = &DecisionEstimateConfig{MaxDecodeContextTokens: 2048}
	c.DecisionPolicy.BudgetRestore = &BudgetRestoreConfig{Guardband: 1}
	// Query-dependent synthetic compute makes zero/partial/full restoration
	// all distinguishable from the fixed per-step fixture used by P4.1.
	curve := []float64{5, 10, 0, 0}
	c.EnginePhases.Compute.PostForward = curve
	c.EnginePhases.Compute.GPUReady = curve
	c.EnginePhases.Compute.OutputReady = curve
	return c
}

func TestBudgetRestoreRunsVariableActionsThroughCommonRuntime(t *testing.T) {
	for _, tc := range []struct{ slo, blocks int64 }{{120, 0}, {180, 1}, {400, 2}} {
		t.Run(fmt.Sprint(tc.slo), func(t *testing.T) {
			c := budgetLabConfig()
			c.Requests[2].TTFTSLOUS = tc.slo
			r, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			var bytes, computed int64
			for _, e := range r.Events {
				if e.Name == "transfer_start" && e.Request == "reader" && e.Destination == "hbm" {
					bytes += e.Bytes
				}
			}
			for _, b := range r.BatchShapes {
				for _, q := range b.Scheduled {
					if q.Request == "reader" {
						computed += q.Query
					}
				}
			}
			if bytes != tc.blocks*r.BlockBytes || computed != max(int64(1), 32-16*tc.blocks) {
				t.Fatalf("budget did not control execution: blocks=%d compute=%d", bytes/r.BlockBytes, computed)
			}
			if len(r.FirstTokenUS) != 3 || r.DecisionEstimateCoverage != RestoreEstimateCoverage {
				t.Fatal("missing completion or coverage")
			}
			for _, d := range r.PolicyDecisions {
				if d.View.Estimates == nil || !d.View.Capabilities.CostEstimates {
					t.Fatal("policy missing prediction evidence")
				}
			}
		})
	}
}

func TestPredictionOnlyDoesNotChangeExecution(t *testing.T) {
	limit := int64(2)
	c := restoreLabConfig(&limit)
	before, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionEstimates = &DecisionEstimateConfig{MaxDecodeContextTokens: 2048}
	after, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before.Events, after.Events) || !reflect.DeepEqual(before.FirstTokenUS, after.FirstTokenUS) || !reflect.DeepEqual(before.BatchShapes, after.BatchShapes) {
		t.Fatal("prediction modified execution")
	}
}

func estimateFixture(t *testing.T) (*restoreEstimator, sim.DecisionView) {
	t.Helper()
	c := budgetLabConfig()
	m, err := newRestoreEstimator(c)
	if err != nil {
		t.Fatal(err)
	}
	v := sim.DecisionView{BlockTokens: 16, MaxBatchTokens: 128, MaxSequences: 2, PrefillChunk: 64, Waiting: []sim.DecisionRequest{{ID: "reader", InputTokens: 32}}, KV: sim.DecisionKVState{PrefixStateKnown: true, TransfersKnown: true, Requests: []sim.DecisionKVRequest{{ID: "reader", RecoverablePrefixBlocks: 2, RecoverablePrefixTokens: 31}}}}
	return m, v
}

func TestRestorePredictionRejectsUnprofiledDecodeHistory(t *testing.T) {
	m, v := estimateFixture(t)
	v.Waiting[0].RecomputeUntilTokens = 35
	v.Waiting[0].EmittedTokens = 3
	for _, computed := range []int64{0, 33} {
		v.Waiting[0].ComputedTokens = computed
		e, err := m.Estimate(v)
		if err != nil || len(e.Requests) != 1 || e.Requests[0].Unavailable != "decode_recompute_not_profiled" || len(e.Requests[0].Choices) != 0 {
			t.Fatal("input-only estimate silently omitted known output recomputation", e, err)
		}
		if err := sim.ValidateDecisionEstimates(v, e); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRestorePredictionAccountsLoadWindowBacklogAndDecode(t *testing.T) {
	m, v := estimateFixture(t)
	before := sim.DecisionView{BlockTokens: v.BlockTokens, MaxBatchTokens: v.MaxBatchTokens, MaxSequences: v.MaxSequences, PrefillChunk: v.PrefillChunk, Waiting: append([]sim.DecisionRequest(nil), v.Waiting...), KV: v.KV}
	a, err := m.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(v, before) {
		t.Fatal("estimator mutated snapshot")
	}
	if err := sim.ValidateDecisionEstimates(v, a); err != nil {
		t.Fatal(err)
	}
	choices := a.Requests[0].Choices
	if len(choices) != 3 || choices[0].ComputeUS != 335 || choices[0].MaxLoadComputeUS != 345 || choices[1].LoadUS != 123 || choices[2].LoadUS != 208 {
		t.Fatalf("unexpected isolated cost envelope: %+v", choices)
	}
	v.KV.PendingTransfers = []sim.DecisionTransfer{{ID: 1, Source: "dram", Destination: "hbm", Bytes: m.phase.BlockBytes}}
	b, err := m.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	if b.Requests[0].Choices[0].LoadUS != 0 || b.Requests[0].Choices[1].LoadUS <= choices[1].LoadUS {
		t.Fatal("public backlog ignored or charged to no-load candidate")
	}
	v.KV.PendingTransfers = nil
	v.Running = []sim.DecisionRequest{{ID: "decode", InputTokens: 16, ComputedTokens: 100}}
	d, err := m.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	if d.Requests[0].Choices[0].ComputeUS <= choices[0].ComputeUS {
		t.Fatal("active decode load ignored")
	}
	v.Running = nil
	m.window = 1
	w, err := m.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	if w.Requests[0].Choices[2].LoadUS <= choices[2].LoadUS {
		t.Fatal("multiple restore submissions ignored")
	}
}

func TestBudgetRestorePressureAndNoFreeDecodeSlot(t *testing.T) {
	c := budgetLabConfig()
	c.Instances[0].HBMBlocks = 10
	c.MaxSequences = 4
	c.PrefillChunk = 16
	c.Requests = nil
	for i := 0; i < 24; i++ {
		tokens := make([]sim.TokenID, 32+16*(i%3))
		for j := range tokens {
			tokens[j] = sim.TokenID(1000*(i%4) + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i * 20), Input: tokens, Output: []sim.TokenID{9, 10, 11, 12}, TTFTSLOUS: 2000})
	}
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.FirstTokenUS) != 24 || len(r.FinishedUS) != 24 {
		t.Fatal("budget policy stranded requests")
	}
	for _, p := range r.Pools {
		if p["reserved"] != 0 || p["read_pins"] != 0 {
			t.Fatal("CPU reservation leaked")
		}
	}
	for _, p := range r.HBM {
		if p["active_or_pinned"] != 0 {
			t.Fatal("HBM reservation leaked")
		}
	}
	m, v := estimateFixture(t)
	v.Running = []sim.DecisionRequest{{ID: "a", InputTokens: 1, ComputedTokens: 1}, {ID: "b", InputTokens: 1, ComputedTokens: 1}}
	e, err := m.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	if e.Requests[0].Unavailable != "no_free_sequence_slot" {
		t.Fatal("full decode occupancy was treated as free compute")
	}
	v.Estimates = &e
	v.Capabilities = sim.DecisionCapabilities{RestoreChoice: true, CostEstimates: true}
	p, _ := sim.NewBudgetRestorePolicy(1, 0)
	if len(p.Decide(v).Restores) != 0 {
		t.Fatal("unknown slot release used as a restore deadline")
	}
}

func TestBudgetRestoreConfigurationRejectsMissingPredictionOrConflicts(t *testing.T) {
	c := budgetLabConfig()
	c.DecisionEstimates = nil
	if err := c.Validate(); err == nil {
		t.Fatal("budget without estimates accepted")
	}
	c = budgetLabConfig()
	c.DecisionPolicy.QueueOrder = "sjf"
	if err := c.Validate(); err == nil {
		t.Fatal("conflicting queue policy accepted")
	}
	c = budgetLabConfig()
	c.DecisionPolicy.BudgetRestore.Guardband = 0
	if err := c.Validate(); err == nil {
		t.Fatal("optimistic/implicit guardband accepted")
	}
}

func TestCompletionBudgetVariantChangesOnlyPolicy(t *testing.T) {
	c := budgetLabConfig()
	c.Requests[2].TTFTSLOUS = 180
	old, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.BudgetRestore.Mode = "completion_bound"
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	blocks := func(result *Result) int64 {
		var bytes int64
		for _, e := range result.Events {
			if e.Name == "transfer_start" && e.Request == "reader" && e.Destination == "hbm" {
				bytes += e.Bytes
			}
		}
		return bytes / result.BlockBytes
	}
	if blocks(old) != 1 || blocks(r) != 2 || r.FirstTokenUS["reader"] >= old.FirstTokenUS["reader"] {
		t.Fatal("policy-only completion bound did not change actual restore/completion")
	}
	if len(r.FinishedUS) != 3 {
		t.Fatal("completion bound stranded requests")
	}
}
