package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func requestSpillConfig(mode string) Config {
	c := budgetLabConfig()
	c.DecisionEstimates = nil
	c.DecisionPolicy = &DecisionPolicyConfig{RequestSpill: true, PrefillPreemption: true, ControlSteps: true, TraceEvents: true,
		PreemptionStorage: &PreemptionStorageConfig{Mode: mode}}
	if mode == "spill" {
		c.DecisionPolicy.PreemptionStorage.Pool = "dram"
	}
	c.Mechanisms.BackgroundStoreMode = "on_preemption"
	c.Policy.Name = "lru_drop"
	c.Instances[0].HBMBlocks = 4
	c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 1, 32, 32
	tokens := func(n, base int) []sim.TokenID {
		r := make([]sim.TokenID, n)
		for i := range r {
			r[i] = sim.TokenID(base + i)
		}
		return r
	}
	c.Requests = []RequestConfig{
		{ID: "victim", At: 0, Input: tokens(64, 1), Output: []sim.TokenID{9}, MaxOutputTokens: 1},
		{ID: "replacement", At: 1, Input: tokens(63, 100), Output: []sim.TokenID{9}, MaxOutputTokens: 1},
	}
	return c
}

func TestRequestSpillPublicFactoryChangesActualStoreRestoreAndCompute(t *testing.T) {
	for _, mode := range []string{"spill", "recompute"} {
		c := requestSpillConfig(mode)
		factories := PolicyFactories{Decision: func(string) sim.DecisionPolicy { return &sim.PrefillSJFPolicy{} }}
		r, err := RunWithPolicies(c, factories)
		if err != nil {
			t.Fatal(err)
		}
		if len(r.FinishedUS) != 2 || r.RequestSpillCostCoverage == "" {
			t.Fatal("missing completion/cost boundary")
		}
		var stored, loaded, computed int64
		for _, e := range r.Events {
			if e.Name == "transfer_start" && e.Request == "victim" {
				if e.Source == "hbm" {
					stored += e.Bytes / r.BlockBytes
				}
				if e.Destination == "hbm" {
					loaded += e.Bytes / r.BlockBytes
				}
			}
		}
		for _, step := range r.BatchShapes {
			for _, work := range step.Scheduled {
				if work.Request == "victim" {
					computed += work.Query
				}
			}
		}
		expectedCompute := int64(96)
		if mode == "spill" {
			expectedCompute = 64
		}
		if computed != expectedCompute || mode == "spill" && (stored != 2 || loaded != 2) || mode == "recompute" && (stored != 0 || loaded != 0) {
			t.Fatalf("%s did not change physical work: store=%d load=%d compute=%d", mode, stored, loaded, computed)
		}
		actions := 0
		for _, record := range r.PolicyDecisions {
			for _, outcome := range record.Feedback.PreemptionStorage {
				actions++
				if outcome.Request != "victim" {
					t.Fatal("storage for an unselected request")
				}
			}
		}
		if actions != 1 {
			t.Fatal("storage action not attached exactly once", actions)
		}
		for _, pool := range r.Pools {
			if pool["reserved"] != 0 || pool["read_pins"] != 0 {
				t.Fatal("leaked pool resources")
			}
		}
		for _, hbm := range r.HBM {
			if hbm["active_or_pinned"] != 0 {
				t.Fatal("leaked HBM resources")
			}
		}
		// An algorithm can emit the same joint actions through the public factory
		// without touching config decoration or the underlying allocator/clock.
		settings := *c.DecisionPolicy.PreemptionStorage
		c.DecisionPolicy.PreemptionStorage = nil
		direct, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy {
			policy, e := settings.NewPolicy(&sim.PrefillSJFPolicy{})
			if e != nil {
				t.Fatal(e)
			}
			return policy
		}})
		if err != nil || !reflect.DeepEqual(r.Events, direct.Events) || !reflect.DeepEqual(r.FirstTokenUS, direct.FirstTokenUS) {
			t.Fatal("configured and directly injected storage policies differ", err)
		}
	}
}

func TestRequestSpillConfigurationRejectsUnsupportedFeesAndTargets(t *testing.T) {
	for _, mutate := range []func(*Config){
		func(c *Config) { c.DecisionPolicy.RequestSpill = false },
		func(c *Config) { c.DecisionPolicy.PrefillPreemption = false },
		func(c *Config) { c.DecisionPolicy.PreemptionStorage.Pool = "missing" },
		func(c *Config) { c.DecisionPolicy.PreemptionStorage.Mode = "bad" },
		func(c *Config) { c.Mechanisms.BackgroundStoreMode = "bad" },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost = &QueueServiceCostConfig{} },
	} {
		c := requestSpillConfig("spill")
		mutate(&c)
		if c.Validate() == nil {
			t.Fatal("unsupported spill configuration accepted")
		}
	}
}

func TestBudgetSpillRunsThroughActualCapacityPressure(t *testing.T) {
	for _, target := range []int64{1000000, 100} {
		c := requestSpillConfig("budget")
		c.DecisionPolicy.PreemptionStorage.Pool = "dram"
		c.DecisionPolicy.BudgetRestore = &BudgetRestoreConfig{Mode: "budget_pressure", Guardband: 1}
		c.DecisionPolicy.CapacityPreemption = true
		c.DecisionPolicy.CapacityReservationView = true
		c.DecodeCapacityReservation = true
		c.DecisionEstimates = &DecisionEstimateConfig{MaxDecodeContextTokens: 2048, Spill: true}
		// Leave tokens for a waiting admission; a 32-token total batch would be
		// exhausted by the running chunk and never trigger physical capacity.
		c.MaxSequences, c.MaxBatchTokens = 2, 48
		c.Requests[0].TTFTSLOUS, c.Requests[1].TTFTSLOUS = target, 1
		r, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		if len(r.FinishedUS) != 2 {
			t.Fatal("budget spill did not finish")
		}
		actions := 0
		for _, record := range r.PolicyDecisions {
			for _, outcome := range record.Feedback.PreemptionStorage {
				actions++
				if outcome.Request != "victim" || target == 1000000 && outcome.NewBlocks != 2 || target == 100 && outcome.Status != "recompute" {
					t.Fatal("budget did not choose the expected physical action", target, outcome)
				}
			}
		}
		if actions != 1 {
			t.Fatal("test did not exercise one real capacity victim", target, actions)
		}
		for _, pool := range r.Pools {
			if pool["reserved"] != 0 || pool["read_pins"] != 0 {
				t.Fatal("spill leaked pool resources")
			}
		}
		for _, hbm := range r.HBM {
			if hbm["active_or_pinned"] != 0 || hbm["decode_reserved_blocks"] != 0 {
				t.Fatal("spill leaked HBM reservations")
			}
		}
	}
}
