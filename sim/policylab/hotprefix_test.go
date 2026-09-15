package policylab

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func hotPrefixLabConfig() Config {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	c.Policy = PolicyConfig{Name: "hotprefix"}
	c.HotPrefix = &kv.HotPrefixConfig{AgingIntervalRequests: 4, AdmissionThreshold: 2, PromotionBlocks: 2, PlannerUS: 3}
	c.Mechanisms.BackgroundStoreMode = "on_reclaim"
	c.PrefixCopySelection = "first_published"
	c.DecisionPolicy = &DecisionPolicyConfig{QueueOrder: "fcfs", ControlSteps: true, TraceEvents: true}
	return c
}

func TestHotPrefixBurstCompletionAndRequestPolicyComposition(t *testing.T) {
	for seed := int64(0); seed < 4; seed++ {
		c := hotPrefixLabConfig()
		c.MaxSequences = 4
		c.MaxBatchTokens = 32
		c.PrefillChunk = 7
		c.Requests = nil
		rng := rand.New(rand.NewSource(seed))
		for i := 0; i < 24; i++ {
			input := make([]sim.TokenID, 17+rng.Intn(65))
			family := rng.Intn(3)
			for j := range input {
				input[j] = sim.TokenID(family*1000 + j)
			}
			c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i/4) * 10000, Input: input, Output: make([]sim.TokenID, 1+rng.Intn(8))})
		}
		p, _ := sim.NewQueueDecisionPolicy("sjf", 0)
		r, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy { return p }})
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if len(r.FinishedUS) != len(c.Requests) || r.HBM["instance_0"]["hotprefix_requests_observed"] != int64(len(c.Requests)) {
			t.Fatal("incomplete requests or double counted retry")
		}
		if r.HBM["instance_0"]["total_refs"] != 0 || r.HBM["instance_0"]["hotprefix_parent_pins"] != 0 || r.Pools["dram"]["reserved"] != 0 || r.Pools["dram"]["read_pins"] != 0 {
			t.Fatal("placement ownership leaked")
		}
		if r.HotPrefixCostCoverage == "" || len(r.InjectedPolicies["decision"]) != 1 {
			t.Fatal("lost policy composition or cost boundary")
		}
	}
}

func TestHotPrefixConfigurationRejectsAmbiguousPlacement(t *testing.T) {
	for _, change := range []func(*Config){
		func(c *Config) { c.HotPrefix = nil },
		func(c *Config) { c.Policy.Name = "lru_drop" },
		func(c *Config) { c.Instances = append(c.Instances, c.Instances[0]) },
		func(c *Config) { c.Mechanisms.BackgroundStoreMode = "continuous" },
		func(c *Config) { c.PromotionControl = true },
		func(c *Config) { c.HotPrefix.AgingIntervalRequests = 0 },
		func(c *Config) { c.HotPrefix.AdmissionThreshold = 256 },
		func(c *Config) { c.Requests[0].ID = "@promotion" },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost = &QueueServiceCostConfig{} },
	} {
		c := hotPrefixLabConfig()
		change(&c)
		if _, err := Run(c); err == nil {
			t.Fatal("unsupported configuration accepted")
		}
	}
	c := hotPrefixLabConfig()
	called := false
	_, err := RunWithPolicies(c, PolicyFactories{Peer: func(string) kv.PeerPolicy { called = true; return kv.BuiltinPeerPolicy{Name: "lru_drop"} }})
	if err == nil || called {
		t.Fatal("conflicting override reached factory")
	}
}

func TestOnReclaimModeDoesNotEagerlyStoreAndHotPrefixIsDeterministic(t *testing.T) {
	c := hotPrefixLabConfig()
	c.Instances[0].HBMBlocks = 128
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if a.Counts["l2_reserve"] != 0 {
		t.Fatal("selective placement silently became continuous write-through")
	}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) {
		t.Fatal("nondeterministic placement")
	}
}
