package policylab

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func nativeLabConfig() Config {
	c := hotPrefixLabConfig()
	c.RequestScheduler = "vllm_native"
	c.RestoreControl = true
	c.DecisionPolicy.ExecutionWork = true
	return c
}

func TestVLLMNativePlacementCompositionAndCompletion(t *testing.T) {
	for _, hot := range []bool{false, true} {
		for seed := int64(0); seed < 3; seed++ {
			c := nativeLabConfig()
			if !hot {
				c.Policy = PolicyConfig{Name: "lru_drop"}
				c.HotPrefix = nil
			}
			c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 4, 16, 7
			c.Requests = nil
			rng := rand.New(rand.NewSource(seed))
			for i := 0; i < 24; i++ {
				input := make([]sim.TokenID, 17+rng.Intn(49))
				family := rng.Intn(3)
				for j := range input {
					input[j] = sim.TokenID(family*1000 + j)
				}
				c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i/4) * 10000, Input: input, Output: make([]sim.TokenID, 1+rng.Intn(20))})
			}
			r, err := Run(c)
			if err != nil {
				t.Fatalf("hot=%v seed=%d: %v", hot, seed, err)
			}
			if len(r.FinishedUS) != len(c.Requests) || len(r.FirstTokenUS) != len(c.Requests) {
				t.Fatal("requests did not complete")
			}
			if r.HBM["instance_0"]["total_refs"] != 0 || r.Pools["dram"]["reserved"] != 0 || r.Pools["dram"]["read_pins"] != 0 {
				t.Fatal("ownership leaked")
			}
			if r.VLLMNativeRevision != sim.VLLMNativeRevision || r.VLLMNativeCostCoverage == "" {
				t.Fatal("missing native provenance/coverage")
			}
		}
	}
}

func TestVLLMNativeRejectsHybridControllersBeforeExecution(t *testing.T) {
	for _, change := range []func(*Config){
		func(c *Config) { c.RestoreControl = false },
		func(c *Config) { c.Instances = append(c.Instances, c.Instances[0]) },
		func(c *Config) { c.DecisionPolicy.QueueOrder = "sjf" },
		func(c *Config) { c.DecisionPolicy.PrefillTokenCap = 1 },
		func(c *Config) { c.DecisionPolicy.AdmissionDelayUS = 1 },
		func(c *Config) { c.DecisionPolicy.PrefillPreemption = true },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost = &QueueServiceCostConfig{} },
	} {
		c := nativeLabConfig()
		change(&c)
		if _, err := Run(c); err == nil {
			t.Fatal("hybrid/unsupported native baseline accepted")
		}
	}
	called := false
	_, err := RunWithPolicies(nativeLabConfig(), PolicyFactories{Decision: func(string) sim.DecisionPolicy { called = true; return nil }})
	if err == nil || called {
		t.Fatal("conflicting factory executed")
	}
}

func TestVLLMNativeWorksWithoutDecisionControllerAndIsDeterministic(t *testing.T) {
	c := nativeLabConfig()
	c.DecisionPolicy = nil
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) || len(a.FinishedUS) != len(c.Requests) {
		t.Fatal("baseline differs or did not drain")
	}
}

func TestVLLMNativeDecodePressurePreservesObservedOutputs(t *testing.T) {
	c := nativeLabConfig()
	c.Policy, c.HotPrefix = PolicyConfig{Name: "lru_drop"}, nil
	c.Instances[0].HBMBlocks = 4
	c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 2, 32, 32
	c.Requests = nil
	for i := 0; i < 2; i++ {
		input := make([]sim.TokenID, 17)
		for j := range input {
			input[j] = sim.TokenID(i*1000 + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), Input: input, Output: make([]sim.TokenID, 24)})
	}
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if r.Metrics.PreemptionCount == 0 || len(r.FinishedUS) != 2 {
		t.Fatal("did not exercise decode preemption and completion")
	}
	last := map[string]int64{}
	replayed := false
	for _, record := range r.PolicyDecisions {
		for _, q := range append(append([]sim.DecisionRequest{}, record.View.Running...), record.View.Waiting...) {
			if q.EmittedTokens < last[q.ID] {
				t.Fatal("preemption erased observed output")
			}
			last[q.ID] = q.EmittedTokens
			if q.EmittedTokens > 0 && q.ComputedTokens == 0 {
				replayed = true
			}
		}
	}
	if !replayed || r.HBM["instance_0"]["total_refs"] != 0 {
		t.Fatal("missing retained output replay or leaked KV")
	}
}
