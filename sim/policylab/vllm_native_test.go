package policylab

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
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

func TestVLLMNativeReclaimSourceReusePolicyMatrix(t *testing.T) {
	var dependencies int64
	for _, capacity := range []int64{8, 12} {
		for _, placement := range []string{"lru_drop", "lfu_store", "hotprefix_2", "prefixkeep_1"} {
			t.Run(fmt.Sprintf("%s/hbm=%d", placement, capacity), func(t *testing.T) {
				c := nativeLabConfig()
				c.StoreSourceReuse, c.TraceBatchShapes = true, true
				c.Instances[0].HBMBlocks = capacity
				c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 3, 16, 7
				if placement == "lru_drop" || placement == "lfu_store" {
					c.Policy, c.HotPrefix = PolicyConfig{Name: placement}, nil
				} else {
					c.HotPrefix.PromotionBlocks = 0
					if placement == "prefixkeep_1" {
						c.HotPrefix.AdmissionThreshold = 1
					}
				}
				c.Requests = nil
				rng := rand.New(rand.NewSource(17))
				for i := 0; i < 24; i++ {
					input := make([]sim.TokenID, 17+rng.Intn(49))
					family := i % 5
					for j := range input {
						input[j] = sim.TokenID(family*1000 + j)
					}
					c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i/4) * 1000, Input: input, Output: make([]sim.TokenID, 1+rng.Intn(20))})
				}
				r, err := Run(c)
				if err != nil {
					t.Fatal(err)
				}
				if len(r.FinishedUS) != 24 || len(r.FirstTokenUS) != 24 || r.StoreSourceReuseContract != kv.StoreSourceReuseContract {
					t.Fatal("incomplete native offloading execution or missing contract version")
				}
				if r.HBM["instance_0"]["total_refs"] != 0 || r.Pools["dram"]["reserved"] != 0 || r.Pools["dram"]["read_pins"] != 0 || r.Counts["transfer_end"] != r.Counts["transfer_adopted"] {
					t.Fatal("STORE/LOAD lifecycle failed to drain")
				}
				ended, fences := map[int64]bool{}, map[int64]bool{}
				for _, e := range r.Events {
					switch e.Name {
					case "hbm_reuse_dependency", "preemption_store_dependency":
						fences[e.Transaction] = true
					case "transfer_end":
						ended[e.Transaction] = true
					case "engine_forward_ready":
						for id := range fences {
							if !ended[id] {
								t.Fatal("forward crossed a live STORE dependency", id, e.Time)
							}
						}
						clear(fences)
					}
				}
				dependencies += r.Counts["hbm_reuse_dependency"]
				if placement == "lfu_store" && r.Counts["hbm_reuse_dependency"] == 0 {
					t.Fatal("fixture did not exercise reclaim source reuse")
				}
			})
		}
	}
	if dependencies == 0 {
		t.Fatal("matrix did not cover execution fences")
	}
}
