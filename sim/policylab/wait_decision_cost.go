package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// WaitDecisionCostConfig prices the additional native policy wrapper only.
// Features are attempts, visible requests, restore choices, all wait updates,
// and pending transfers. A superseded attempt still consumes its full service.
type WaitDecisionCostConfig struct {
	Family          string              `json:"family"`
	RatesUS         []int64             `json:"rates_us"`
	MaxCounts       []int64             `json:"max_counts"`
	MaxInputTokens  int64               `json:"max_input_tokens"`
	MaxOutputTokens int                 `json:"max_output_tokens"`
	HBMBlocks       int64               `json:"hbm_blocks"`
	Shape           RestoreServiceShape `json:"shape"`
	Provenance      string              `json:"provenance"`
}

type waitDecisionCost struct{ config WaitDecisionCostConfig }

func newWaitDecisionCost(c Config) (*waitDecisionCost, error) {
	if c.DecisionPolicy == nil || c.DecisionPolicy.WaitDecisionCost == nil || c.DecisionPolicy.ExtraCost != nil || c.DecisionPolicy.RestoreServiceCost != nil {
		return nil, fmt.Errorf("wait decision profile requires no other decision cost profile")
	}
	p := *c.DecisionPolicy.WaitDecisionCost
	if err := validateWaitServiceScope(c, p.Family, p.Shape, p.HBMBlocks, p.MaxInputTokens, p.MaxOutputTokens, p.Provenance); err != nil {
		return nil, err
	}
	if c.DecisionPolicy.PrefixSharing || c.DecisionPolicy.ProducerPrefixes || c.DecisionPolicy.ProducerWait != nil || c.DecisionPolicy.BalancedBatch != nil || c.DecisionPolicy.PrefillPreemption && p.Family != "capacity" {
		return nil, fmt.Errorf("wait decision profile does not cover additional sharing, producer, balanced or explicit preemption work")
	}
	if len(p.RatesUS) != 5 || len(p.MaxCounts) != 5 || p.MaxCounts[0] != 1 {
		return nil, fmt.Errorf("wait decision profile requires five features and one attempt per service")
	}
	for _, v := range append(append([]int64(nil), p.RatesUS...), p.MaxCounts...) {
		if v < 0 {
			return nil, fmt.Errorf("negative wait decision coefficient or bound")
		}
	}
	p.RatesUS = append([]int64(nil), p.RatesUS...)
	p.MaxCounts = append([]int64(nil), p.MaxCounts...)
	return &waitDecisionCost{p}, nil
}

// WaitDecisionCounts uses only the current detached view and proposed plan.
// The native wait_updates feature includes waiting and resident requests, and
// explicit deadline clears; it is not LinearDecisionCost.PerPrefillWaitUS.
func WaitDecisionCounts(v sim.DecisionView, plan sim.DecisionPlan) ([]int64, error) {
	if !v.Capabilities.PrefillWaits || !v.KV.TransfersKnown {
		return nil, fmt.Errorf("wait decision requires known waits and pending transfers")
	}
	var choices int64
	if v.Estimates != nil {
		for _, r := range v.Estimates.Requests {
			choices += int64(len(r.Choices))
		}
	}
	return []int64{1, int64(len(v.Waiting) + len(v.Running)), choices, int64(len(plan.Waits)), int64(len(v.KV.PendingTransfers))}, nil
}

func (m *waitDecisionCost) Estimate(v sim.DecisionView, plan sim.DecisionPlan) (sim.DecisionCostEstimate, error) {
	p := m.config
	if v.Capabilities.CapacityPreemption != (p.Family == "capacity") || v.HBMCapacityBlocks != p.HBMBlocks || len(plan.Promotions) > 0 || len(plan.Preemptions) > 0 {
		return sim.DecisionCostEstimate{}, fmt.Errorf("decision is outside wait profile family or action coverage")
	}
	for _, group := range [][]sim.DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			if r.InputTokens > p.MaxInputTokens || r.ClientOutputLimit <= 0 || r.ClientOutputLimit > p.MaxOutputTokens || r.RecomputeUntilTokens > r.InputTokens {
				return sim.DecisionCostEstimate{}, fmt.Errorf("decision request is outside wait profile input, output or recompute coverage")
			}
		}
	}
	counts, err := WaitDecisionCounts(v, plan)
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	var total int64
	for i, n := range counts {
		if n > p.MaxCounts[i] {
			return sim.DecisionCostEstimate{}, fmt.Errorf("wait decision feature %d exceeds profile coverage: %d", i, n)
		}
		if n > 0 && p.RatesUS[i] > (math.MaxInt64-total)/n {
			return sim.DecisionCostEstimate{}, fmt.Errorf("wait decision fee overflow")
		}
		total += n * p.RatesUS[i]
	}
	return sim.DecisionCostEstimate{ExtraUS: total, Provenance: p.Provenance}, nil
}
