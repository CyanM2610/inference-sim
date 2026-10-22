package policylab

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/latency"
)

func phaseTestConfig(c *Config) {
	constant := func(n float64) []float64 { return []float64{n, 0, 0, 0} }
	b, _ := latency.KVBytesPerToken(c.Model, 1)
	c.BatchCost = &BatchCostConfig{CoefficientsUS: constant(100), Provenance: "synthetic behavioral fixture"}
	c.EnginePhases = &EnginePhaseCostConfig{Compute: EnginePhaseCurve{PreForward: constant(5), PostForward: constant(60), GPUReady: constant(100), OutputReady: constant(100), Poll: constant(5), Tail: constant(5)},
		Idle: kv.EngineStepTiming{PreForwardUS: 5, PostForwardUS: 10, PollUS: 5, TailUS: 5}, LoadSubmit: []float64{5, 1}, StoreSubmit: []float64{5, 1}, BlockBytes: int64(b * float64(c.BlockTokens)), Provenance: "synthetic behavioral fixture"}
	c.Mechanisms = &kv.PeerMechanisms{BackgroundStorePool: "dram", GroupTransfers: true, ConcurrentRestores: true, RestoreWindow: 32}
}

func TestEnginePhaseIdleDrainAndPressureProgress(t *testing.T) {
	var idleSteps int64
	for seed := int64(0); seed < 8; seed++ {
		for _, policy := range []string{"lru_drop", "lfu_store"} {
			c := labConfig(1, "dram")
			phaseTestConfig(&c)
			c.Policy.Name = policy
			c.MaxSequences = 4
			c.PrefillChunk = 32
			c.Requests = nil
			rng := rand.New(rand.NewSource(seed))
			for i := 0; i < 60; i++ {
				tokens := make([]sim.TokenID, 17+rng.Intn(65))
				group := rng.Intn(5)
				for j := range tokens {
					tokens[j] = sim.TokenID(group*1000 + j)
				}
				c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i * 20), Input: tokens, Output: make([]sim.TokenID, 1+rng.Intn(12))})
			}
			r, err := Run(c)
			if err != nil {
				t.Fatalf("seed %d policy %s: %v HBM=%v", seed, policy, err, r.HBM)
			}
			idleSteps += r.Counts["sim.EngineIdleCompleteEvent"]
			if r.Counts["engine_worker_poll"] == 0 || r.Counts["transfer_adopted"] == 0 {
				t.Fatal("fixture missed control-only drain/transfer adoption")
			}
			for _, pool := range r.Pools {
				if pool["reserved"] != 0 || pool["read_pins"] != 0 {
					t.Fatal("phase pipeline did not drain")
				}
			}
			if r.Counts["transfer_end"] != r.Counts["transfer_adopted"] {
				t.Fatal("physical work was lost or adopted twice")
			}
		}
	}
	if idleSteps == 0 {
		t.Fatal("pressure matrix did not exercise control-only drain")
	}
}

func TestEnginePhaseRejectsUncalibratedBlockSize(t *testing.T) {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	c.EnginePhases.BlockBytes++
	if _, err := Run(c); err == nil {
		t.Fatal("accepted a different KV block layout")
	}
}

func TestSingleTokenPhaseChargesActualExecution(t *testing.T) {
	for _, input := range [][]sim.TokenID{{1}, {1, 2}} {
		c := labConfig(1, "dram")
		phaseTestConfig(&c)
		c.Requests = []RequestConfig{{ID: "r", Input: input, Output: []sim.TokenID{3}}}
		base, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		x := c.EnginePhases.Compute
		x.GPUReady, x.OutputReady = []float64{200, 0, 0, 0}, []float64{200, 0, 0, 0}
		c.EnginePhases.SingleTokenCompute = &x
		changed, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		want := int64(0)
		if len(input) == 1 {
			want = 100
		}
		if got := changed.FirstTokenUS["r"] - base.FirstTokenUS["r"]; got != want {
			t.Fatalf("input=%v: actual first token changed by %d, want %d", input, got, want)
		}
	}
}

func TestSingleTokenPhasePreservesMixedAndIdlePaths(t *testing.T) {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	base := *c.EnginePhases
	x := base.Compute
	x.OutputReady = []float64{200, 0, 0, 0}
	c.EnginePhases.SingleTokenCompute = &x
	for _, work := range [][]sim.BatchWork{nil, {{NewTokens: 1}, {NewTokens: 16}}} {
		if c.EnginePhases.PredictEngineStep(work) != base.PredictEngineStep(work) {
			t.Fatal("specialization changed mixed-query or empty step")
		}
	}
	if c.EnginePhases.PredictEngineStep([]sim.BatchWork{{NewTokens: 1}, {NewTokens: 1}}).OutputReadyUS != 200 {
		t.Fatal("homogeneous one-token batch did not use the measured specialization")
	}
	x.OutputReady = []float64{math.NaN(), 0, 0, 0}
	if err := c.EnginePhases.Validate(); err == nil {
		t.Fatal("accepted invalid optional curve")
	}
}
