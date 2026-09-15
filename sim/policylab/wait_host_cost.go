package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// CompletionRatesUS prices additional controller updates/grants/free calls.
// RegistrationUS is the complete measured registration service and REPLACES
// external enqueue. Nil stages are disabled; zero is an explicit observation.
type WaitHostServiceCostConfig struct {
	Family              string              `json:"family"`
	CompletionRatesUS   []int64             `json:"completion_rates_us,omitempty"`
	MaxCompletionCounts []int64             `json:"max_completion_counts,omitempty"`
	RegistrationUS      *int64              `json:"registration_us,omitempty"`
	MaxInputTokens      int64               `json:"max_input_tokens"`
	MaxOutputTokens     int                 `json:"max_output_tokens"`
	HBMBlocks           int64               `json:"hbm_blocks"`
	Shape               RestoreServiceShape `json:"shape"`
	Provenance          string              `json:"provenance"`
}

type waitHostServiceCost struct{ config WaitHostServiceCostConfig }

func newWaitHostServiceCost(c Config) (*waitHostServiceCost, error) {
	if c.DecisionPolicy == nil || c.DecisionPolicy.WaitHostServiceCost == nil || c.DecisionPolicy.HostServiceCost != nil {
		return nil, fmt.Errorf("wait host profile cannot share a boundary with an existing host profile")
	}
	p := *c.DecisionPolicy.WaitHostServiceCost
	if err := validateWaitServiceScope(c, p.Family, p.Shape, p.HBMBlocks, p.MaxInputTokens, p.MaxOutputTokens, p.Provenance); err != nil {
		return nil, err
	}
	if p.CompletionRatesUS == nil && p.RegistrationUS == nil {
		return nil, fmt.Errorf("wait host profile has no enabled stage")
	}
	if p.RegistrationUS != nil && (*p.RegistrationUS < 0 || c.BatchCost == nil || c.BatchCost.EnqueueUS != 0) {
		return nil, fmt.Errorf("complete wait registration requires nonnegative service and zero external enqueue cost")
	}
	if p.CompletionRatesUS != nil && (len(p.CompletionRatesUS) != 3 || len(p.MaxCompletionCounts) != 3) {
		return nil, fmt.Errorf("wait completion requires three rates and bounds")
	}
	for _, n := range append(append([]int64(nil), p.CompletionRatesUS...), p.MaxCompletionCounts...) {
		if n < 0 {
			return nil, fmt.Errorf("negative wait host coefficient or bound")
		}
	}
	p.CompletionRatesUS = append([]int64(nil), p.CompletionRatesUS...)
	p.MaxCompletionCounts = append([]int64(nil), p.MaxCompletionCounts...)
	if p.RegistrationUS != nil {
		value := *p.RegistrationUS
		p.RegistrationUS = &value
	}
	return &waitHostServiceCost{p}, nil
}

// The basic native adapter has no additional capacity-update/free wrapper.
// Logical finishing requests are retained in HostServiceWork regardless; zero
// additional wrapper counts do not mean the native engine's cleanup is free.
func WaitCompletionCounts(w sim.HostServiceWork, family string) ([]int64, error) {
	if w.Stage != "completion" || (family != "basic" && family != "capacity") {
		return nil, fmt.Errorf("unsupported wait completion boundary")
	}
	seen := map[string]bool{}
	for _, id := range w.Requests {
		if seen[id] {
			return nil, fmt.Errorf("duplicate completion request")
		}
		seen[id] = true
	}
	for _, id := range w.Finishing {
		if !seen[id] {
			return nil, fmt.Errorf("invalid or duplicate finishing request")
		}
		delete(seen, id)
	}
	counts := []int64{0, int64(len(w.Requests)), 0}
	if family == "capacity" {
		counts[0] = 1
		counts[2] = int64(len(w.Finishing))
	}
	return counts, nil
}

func (m *waitHostServiceCost) EstimateHostService(w sim.HostServiceWork) (sim.DecisionCostEstimate, error) {
	p := m.config
	var total int64
	switch w.Stage {
	case "registration":
		if p.RegistrationUS == nil || len(w.Requests) != 1 || len(w.Finishing) != 0 {
			return sim.DecisionCostEstimate{}, fmt.Errorf("unsupported wait registration work")
		}
		total = *p.RegistrationUS
	case "completion":
		if len(p.CompletionRatesUS) != 3 {
			return sim.DecisionCostEstimate{}, fmt.Errorf("wait completion stage disabled")
		}
		counts, err := WaitCompletionCounts(w, p.Family)
		if err != nil {
			return sim.DecisionCostEstimate{}, err
		}
		for i, n := range counts {
			if n > p.MaxCompletionCounts[i] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("wait completion feature %d exceeds profile coverage: %d", i, n)
			}
			rate := p.CompletionRatesUS[i]
			if n > 0 && rate > (math.MaxInt64-total)/n {
				return sim.DecisionCostEstimate{}, fmt.Errorf("wait completion fee overflow")
			}
			total += rate * n
		}
	default:
		return sim.DecisionCostEstimate{}, fmt.Errorf("unknown wait host stage %q", w.Stage)
	}
	return sim.DecisionCostEstimate{ExtraUS: total, Provenance: p.Provenance}, nil
}
