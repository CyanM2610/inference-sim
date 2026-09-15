package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// The normal source-locked driver has two bookkeeping statements and one
// output-accounting loop after each observed engine return. Optional loop rates
// add the preceding condition/bookkeeping/arrival-loop work. Driver/observer
// residuals remain outside this component.
type WaitPostStepCostConfig struct {
	PreStepRatesUS     []int64             `json:"pre_step_rates_us,omitempty"`
	Family             string              `json:"family"`
	BookkeepingUS      int64               `json:"bookkeeping_us"`
	OutputAccountingUS int64               `json:"output_accounting_us"`
	IdleRatesUS        []int64             `json:"idle_rates_us"`
	MaxIdleCounts      []int64             `json:"max_idle_counts"`
	MaxInputTokens     int64               `json:"max_input_tokens"`
	MaxOutputTokens    int                 `json:"max_output_tokens"`
	HBMBlocks          int64               `json:"hbm_blocks"`
	Shape              RestoreServiceShape `json:"shape"`
	Provenance         string              `json:"provenance"`
}

type waitPostStepCost struct{ config WaitPostStepCostConfig }

func newWaitPostStepCost(c Config) (*waitPostStepCost, error) {
	if c.DecisionPolicy == nil || c.DecisionPolicy.WaitPostStepCost == nil {
		return nil, fmt.Errorf("missing wait post-step profile")
	}
	p := *c.DecisionPolicy.WaitPostStepCost
	if p.PreStepRatesUS != nil && len(p.PreStepRatesUS) != 3 {
		return nil, fmt.Errorf("pre-step driver requires condition, bookkeeping and arrivals rates")
	}
	if err := validateWaitServiceScope(c, p.Family, p.Shape, p.HBMBlocks, p.MaxInputTokens, p.MaxOutputTokens, p.Provenance); err != nil {
		return nil, err
	}
	if p.BookkeepingUS < 0 || p.OutputAccountingUS < 0 || len(p.IdleRatesUS) != 3 || len(p.MaxIdleCounts) != 3 || p.MaxIdleCounts[0] != 1 {
		return nil, fmt.Errorf("invalid post-step rates or idle coverage")
	}
	for _, n := range append(append([]int64(nil), p.IdleRatesUS...), p.MaxIdleCounts...) {
		if n < 0 {
			return nil, fmt.Errorf("negative post-step rate or bound")
		}
	}
	for _, n := range p.PreStepRatesUS {
		if n < 0 {
			return nil, fmt.Errorf("negative pre-step driver rate")
		}
	}
	p.PreStepRatesUS = append([]int64(nil), p.PreStepRatesUS...)
	p.IdleRatesUS = append([]int64(nil), p.IdleRatesUS...)
	p.MaxIdleCounts = append([]int64(nil), p.MaxIdleCounts...)
	return &waitPostStepCost{p}, nil
}

func (m *waitPostStepCost) DriverLoopCostsEnabled() bool { return m.config.PreStepRatesUS != nil }

func (m *waitPostStepCost) EstimatePostStep(w sim.PostStepWork) (sim.DecisionCostEstimate, error) {
	p := m.config
	var rates, counts []int64
	switch w.Stage {
	case "before_step", "loop_exit":
		if len(p.PreStepRatesUS) != 3 {
			return sim.DecisionCostEstimate{}, fmt.Errorf("pre-step driver costs are disabled")
		}
		rates, counts = p.PreStepRatesUS, []int64{1, 1, 1}
		if w.Stage == "loop_exit" {
			counts = []int64{1, 0, 0}
		}
	case "after_output":
		rates, counts = []int64{p.BookkeepingUS, p.OutputAccountingUS}, []int64{2, 1}
	case "idle_check":
		rates, counts = p.IdleRatesUS, []int64{1, int64(len(w.Requests)), int64(len(w.Held))}
		seen := map[string]bool{}
		for _, id := range w.Requests {
			if seen[id] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("duplicate live request at idle check")
			}
			seen[id] = true
		}
		for _, id := range w.Held {
			if !seen[id] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("held request is not a distinct live request")
			}
			delete(seen, id)
		}
		for i, n := range counts {
			if n > p.MaxIdleCounts[i] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("idle feature %d exceeds profile coverage: %d", i, n)
			}
		}
	default:
		return sim.DecisionCostEstimate{}, fmt.Errorf("unsupported post-step stage %q", w.Stage)
	}
	var total int64
	for i, n := range counts {
		if n > 0 && rates[i] > (math.MaxInt64-total)/n {
			return sim.DecisionCostEstimate{}, fmt.Errorf("post-step fee overflow")
		}
		total += rates[i] * n
	}
	return sim.DecisionCostEstimate{ExtraUS: total, Provenance: p.Provenance}, nil
}
