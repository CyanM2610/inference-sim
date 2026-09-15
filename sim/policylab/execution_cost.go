package policylab

import (
	"fmt"
	"math"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

// CapacityExecutionCostConfig declares a partial profile of the native capacity
// adapter. Rates price [constant, allocation attempts, failed reservations,
// actual preemptions, restore matches]; ActionUS prices native preempt/free once.
// Other stages and worker reset remain separate coverage obligations.
type CapacityExecutionCostConfig struct {
	MaxInputTokens  int64               `json:"max_input_tokens"`
	MaxOutputTokens int                 `json:"max_output_tokens"`
	HBMBlocks       int64               `json:"hbm_blocks"`
	RatesUS         []int64             `json:"rates_us"`
	ActionUS        int64               `json:"action_us"`
	MaxCounts       []int64             `json:"max_counts"`
	Shape           RestoreServiceShape `json:"shape"`
	Provenance      string              `json:"provenance"`
}

type capacityExecutionCost struct{ config CapacityExecutionCostConfig }

func newCapacityExecutionCost(c Config) (*capacityExecutionCost, error) {
	if c.DecisionPolicy == nil || c.DecisionPolicy.ExecutionCost == nil || !c.DecisionPolicy.CapacityPreemption || c.EnginePhases == nil || c.BatchPrefixReuse || c.PromotionControl {
		return nil, fmt.Errorf("capacity execution cost requires the single-instance capacity engine-phase path")
	}
	p := *c.DecisionPolicy.ExecutionCost
	if p.MaxInputTokens <= 0 || p.MaxOutputTokens <= 0 || len(c.Instances) != 1 || p.HBMBlocks != c.Instances[0].HBMBlocks {
		return nil, fmt.Errorf("execution profile requires matching HBM geometry and positive input/output bounds")
	}
	for _, r := range c.Requests {
		if int64(len(r.Input)) > p.MaxInputTokens || r.MaxOutputTokens <= 0 || r.MaxOutputTokens > p.MaxOutputTokens {
			return nil, fmt.Errorf("request exceeds execution profile input/output coverage")
		}
	}
	if len(p.RatesUS) != 5 || len(p.MaxCounts) != 4 || p.ActionUS < 0 || strings.TrimSpace(p.Provenance) == "" || p.Shape != serviceShape(c) {
		return nil, fmt.Errorf("invalid capacity execution rates, limits, provenance or geometry")
	}
	for _, rates := range [][]int64{p.RatesUS, p.MaxCounts} {
		for _, x := range rates {
			if x < 0 {
				return nil, fmt.Errorf("negative execution coefficient or limit")
			}
		}
	}
	p.RatesUS = append([]int64(nil), p.RatesUS...)
	p.MaxCounts = append([]int64(nil), p.MaxCounts...)
	return &capacityExecutionCost{config: p}, nil
}

func (m *capacityExecutionCost) EstimateExecution(w sim.BatchExecutionWork, preemptions int64) (sim.DecisionCostEstimate, error) {
	counts := []int64{int64(len(w.Allocations)), 0, preemptions, 0}
	for _, a := range w.Allocations {
		if a.Granted {
			if a.Failure != nil {
				return sim.DecisionCostEstimate{}, fmt.Errorf("stale failure on accepted allocation")
			}
			continue
		}
		if a.Failure == nil || a.Failure.Request != a.Request {
			return sim.DecisionCostEstimate{}, fmt.Errorf("unknown allocation outcome in capacity cost")
		}
		switch {
		case a.Failure.Kind == "capacity":
			counts[1]++
		case a.Failure.Kind == "wait" && a.Failure.Reason == "restore_submitted":
			// The native allocator accepted load targets; the simulator returns
			// false only to defer compute until those loads are adopted.
		default:
			return sim.DecisionCostEstimate{}, fmt.Errorf("capacity execution profile does not cover allocation wait %s/%s", a.Failure.Kind, a.Failure.Reason)
		}
	}
	for _, p := range w.Prefixes {
		if !p.AdoptingRestore {
			counts[3]++
		}
	}
	for i, n := range counts {
		if n < 0 || n > m.config.MaxCounts[i] {
			return sim.DecisionCostEstimate{}, fmt.Errorf("execution feature %d exceeds declared profile coverage: %d", i, n)
		}
	}
	extra := m.config.RatesUS[0]
	for i, n := range counts {
		rate := m.config.RatesUS[i+1]
		if n > 0 && rate > (math.MaxInt64-extra)/n {
			return sim.DecisionCostEstimate{}, fmt.Errorf("execution cost overflow")
		}
		extra += rate * n
	}
	if preemptions > 0 && m.config.ActionUS > (math.MaxInt64-extra)/preemptions {
		return sim.DecisionCostEstimate{}, fmt.Errorf("action cost overflow")
	}
	extra += m.config.ActionUS * preemptions
	return sim.DecisionCostEstimate{ExtraUS: extra, Provenance: m.config.Provenance}, nil
}
