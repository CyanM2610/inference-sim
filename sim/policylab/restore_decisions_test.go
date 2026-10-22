package policylab

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type boundedRestorePolicy struct {
	*sim.QueueDecisionPolicy
	t     *testing.T
	steps int
	last  []sim.DecisionView
}

func (p *boundedRestorePolicy) Decide(v sim.DecisionView) sim.DecisionPlan {
	p.steps++
	p.last = append(p.last, v)
	if len(p.last) > 12 {
		p.last = p.last[1:]
	}
	// A small fixed workload must either finish or report its repeating state;
	// an allocator regression should not spin until the Go suite timeout.
	if p.steps > 1000 {
		data, _ := json.Marshal(p.last)
		p.t.Fatalf("restore limit %d exceeded pressure progress bound: %s", *p.RestorePrefixBlocks, data)
	}
	return p.QueueDecisionPolicy.Decide(v)
}

func restoreLabConfig(limit *int64) Config {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	c.TraceBatchShapes = true
	c.RestoreControl, c.StoreSourceReuse = true, true
	c.DecisionPolicy = &DecisionPolicyConfig{RestorePrefixBlocks: limit, TraceEvents: true}
	c.Instances[0].HBMBlocks = 4
	c.Pools[0].CapacityBlocks = 16
	tokens := func(n, base int) []sim.TokenID {
		out := make([]sim.TokenID, n)
		for i := range out {
			out[i] = sim.TokenID(base + i)
		}
		return out
	}
	c.Requests = []RequestConfig{
		{ID: "warm", At: 0, Input: tokens(32, 1), Output: []sim.TokenID{9}},
		{ID: "evict", At: 100000, Input: tokens(64, 100), Output: []sim.TokenID{9}},
		{ID: "reader", At: 200000, Input: tokens(32, 1), Output: []sim.TokenID{9}},
	}
	return c
}

func TestRestorePolicyConfigChangesRealCopyAndBatchWork(t *testing.T) {
	for _, limit := range []int64{0, 1, 2, 999} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			c := restoreLabConfig(&limit)
			r, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			var copied, computed int64
			for _, e := range r.Events {
				if e.Name == "transfer_start" && e.Request == "reader" && e.Destination == "hbm" {
					copied += e.Bytes
				}
			}
			for _, batch := range r.BatchShapes {
				for _, w := range batch.Scheduled {
					if w.Request == "reader" {
						computed += w.Query
					}
				}
			}
			blocks := min(limit, int64(2))
			if copied != blocks*r.BlockBytes || computed != max(int64(1), 32-blocks*16) {
				t.Fatalf("loading and recompute did not follow policy: bytes=%d computed=%d", copied, computed)
			}
			if len(r.FirstTokenUS) != 3 || len(r.FinishedUS) != 3 || r.FirstTokenUS["reader"] <= 200000 {
				t.Fatal("missing or fabricated output")
			}
			var initial, feedback bool
			for _, record := range r.PolicyDecisions {
				for _, q := range record.View.KV.Requests {
					if q.ID == "reader" && q.LocalPrefixBlocks == 0 && q.RecoverablePrefixBlocks == 2 {
						initial = true
					}
				}
				for _, u := range record.Feedback.RestoreUpdates {
					if u.Request == "reader" && u.MaxPrefixBlocks == blocks {
						feedback = true
					}
				}
			}
			if !initial || !feedback {
				t.Fatal("initial cache equality or action feedback missing")
			}
			for _, pool := range r.Pools {
				if pool["reserved"] != 0 || pool["read_pins"] != 0 {
					t.Fatal("pool lease leaked")
				}
			}
			for _, hbm := range r.HBM {
				if hbm["active_or_pinned"] != 0 {
					t.Fatal("HBM lease leaked")
				}
			}
		})
	}
}

func TestRestoreControlRequiresSupportedBackend(t *testing.T) {
	zero := int64(0)
	c := restoreLabConfig(&zero)
	c.RestoreControl = false
	if err := c.Validate(); err == nil {
		t.Fatal("action accepted without mechanism")
	}
	c.RestoreControl = true
	c.EnginePhases = nil
	if err := c.Validate(); err == nil {
		t.Fatal("unsupported backend accepted")
	}
	c = restoreLabConfig(&zero)
	c.Mechanisms.ConcurrentRestores = false
	if err := c.Validate(); err == nil {
		t.Fatal("serialized restore accepted")
	}
}

func TestRestoreControlConcurrentPressureDrains(t *testing.T) {
	for _, limit := range []int64{0, 1, 2, 8} {
		c := restoreLabConfig(&limit)
		c.Instances[0].HBMBlocks = 10
		c.MaxSequences = 4
		c.PrefillChunk = 16
		c.Requests = nil
		for i := 0; i < 36; i++ {
			input := make([]sim.TokenID, 32+16*(i%3))
			for j := range input {
				input[j] = sim.TokenID(1000*(i%4) + j)
			}
			c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i * 20), Input: input, Output: []sim.TokenID{9, 10, 11}})
		}
		r, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy {
			p, _ := sim.NewQueueDecisionPolicy("fcfs", 0)
			p.RestorePrefixBlocks = &limit
			return &boundedRestorePolicy{QueueDecisionPolicy: p, t: t}
		}})
		if err != nil {
			t.Fatalf("limit %d: %v", limit, err)
		}
		if len(r.FirstTokenUS) != 36 || len(r.FinishedUS) != 36 || r.Counts["transfer_end"] != r.Counts["transfer_adopted"] {
			t.Fatal("requests/transfers did not drain")
		}
		observed := map[string]int64{}
		history := false
		for _, record := range r.PolicyDecisions {
			for _, request := range append(append([]sim.DecisionRequest(nil), record.View.Waiting...), record.View.Running...) {
				if request.EmittedTokens < observed[request.ID] || request.EmittedTokens > 3 {
					t.Fatal("preemption lost or duplicated generated output", request)
				}
				observed[request.ID] = request.EmittedTokens
				history = history || request.RecomputeUntilTokens > request.InputTokens
			}
		}
		if history && r.RequestRecomputeCostCoverage == "" {
			t.Fatal("recompute CPU coverage was not disclosed")
		}
		for _, p := range r.Pools {
			if p["reserved"] != 0 || p["read_pins"] != 0 {
				t.Fatal("pressure leaked CPU reservation")
			}
		}
		for _, p := range r.HBM {
			if p["active_or_pinned"] != 0 {
				t.Fatal("pressure leaked HBM reservation")
			}
		}
	}
}
