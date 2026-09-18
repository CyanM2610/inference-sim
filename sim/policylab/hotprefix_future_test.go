package policylab

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func futureLabConfig(mode string) Config {
	c := nativeLabConfig()
	zero := int64(0)
	c.HotPrefix = &kv.HotPrefixConfig{AgingIntervalRequests: 16, HBMEvictionUnit: "logical_segment", HBMScore: mode, ShadowTTLUS: &zero, Diagnostics: &kv.HotPrefixDiagnosticsConfig{TraceCandidates: true}}
	c.AllowFuturePlacement = mode == "oracle_next_arrival" || mode == "oracle_remaining"
	c.Instances[0].HBMBlocks = 10
	c.Pools[0].CapacityBlocks = 64
	c.MaxSequences = 2
	c.MaxBatchTokens = 64
	c.PrefillChunk = 32
	c.StoreSourceReuse = true
	c.TraceBatchShapes = true
	c.Requests = nil
	for i, f := range []int{0, 1, 2, 0, 3, 1, 0, 2, 0, 1, 3, 0} {
		input := make([]sim.TokenID, 65)
		for j := range input {
			input[j] = sim.TokenID(f*1000 + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i) * 100000, Input: input, Output: []sim.TokenID{9, 8}})
	}
	return c
}

func TestFuturePlacementRequiresExplicitIsolation(t *testing.T) {
	for _, change := range []func(*Config){
		func(c *Config) { c.AllowFuturePlacement = false },
		func(c *Config) { c.HotPrefix.HBMScore = "paper" },
		func(c *Config) { c.HotPrefix = nil; c.Policy.Name = "lru_drop" },
		func(c *Config) { c.Requests[1].AfterRequest = c.Requests[0].ID; c.Requests[1].At = 0 },
		func(c *Config) { c.HotPrefix.AdmissionThreshold = 2 },
	} {
		c := futureLabConfig("oracle_next_arrival")
		change(&c)
		if _, err := Run(c); err == nil {
			t.Fatal("ambiguous future policy accepted")
		}
	}
}

func TestFuturePlacementRunsRealEngineAndClosedLoop(t *testing.T) {
	for _, mode := range []string{"oracle_next_arrival", "oracle_remaining"} {
		c := futureLabConfig(mode)
		if mode == "oracle_remaining" {
			for i := 1; i < len(c.Requests); i++ {
				c.Requests[i].AfterRequest = c.Requests[i-1].ID
				c.Requests[i].At = 0
				c.Requests[i].ThinkTimeUS = 100000
			}
		}
		a, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		b, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a.Events, b.Events) || a.FuturePlacementCoverage == "" || len(a.FinishedUS) != len(c.Requests) {
			t.Fatal("future execution incomplete or nondeterministic")
		}
		if a.HBM["instance_0"]["total_refs"] != 0 || a.Pools["dram"]["reserved"] != 0 || a.Pools["dram"]["read_pins"] != 0 {
			t.Fatal("oracle bypassed ownership lifecycle")
		}
		if a.HotPrefixDiagnostics["instance_0"].Choices == 0 {
			t.Fatal("no actual choices exercised")
		}
		for _, r := range c.Requests {
			if r.AfterRequest != "" && a.ArrivalUS[r.ID] != a.FinishedUS[r.AfterRequest]+r.ThinkTimeUS {
				t.Fatal("oracle froze endogenous arrival")
			}
		}
	}
}

func TestOnlinePlacementUnaffectedByUnarrivedContent(t *testing.T) {
	c := futureLabConfig("lru")
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	cut := c.Requests[len(c.Requests)-1].At
	for j := range c.Requests[len(c.Requests)-1].Input {
		c.Requests[len(c.Requests)-1].Input[j] += 5000
	}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	var left, right []kv.PeerRecord
	for _, e := range a.Events {
		if e.Time < cut {
			left = append(left, e)
		}
	}
	for _, e := range b.Events {
		if e.Time < cut {
			right = append(right, e)
		}
	}
	if !reflect.DeepEqual(left, right) || a.FuturePlacementCoverage != "" {
		t.Fatal("future content entered online execution")
	}
}
