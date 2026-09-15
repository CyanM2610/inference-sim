package policylab

import (
	"fmt"
	"math"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

var restoreWorkNames = []string{"fixed", "visible_requests", "waiting_requests", "pending_transfers", "prompt_blocks", "candidates", "compute_steps", "compute_items", "load_windows"}

// RestoreServiceShape identifies the predictor and native adapter geometry of a
// measured CPU service profile. A new execution shape needs its own validation.
type RestoreServiceShape struct {
	BlockTokens         int64 `json:"block_tokens"`
	MaxBatchTokens      int64 `json:"max_batch_tokens"`
	MaxSequences        int64 `json:"max_sequences"`
	PrefillChunk        int64 `json:"prefill_chunk"`
	PrefillTokenCap     int64 `json:"prefill_token_cap"`
	RestoreWindowBlocks int64 `json:"restore_window_blocks"`
	MaxInputTokens      int64 `json:"max_input_tokens"`
}

// RestoreServiceCostConfig contains independently profiled CPU operation rates.
// It does not calibrate GPU work or silently treat unmeasured host work as free:
// Coverage must describe exclusions. MaxWork rejects unmeasured extrapolation.
type RestoreServiceCostConfig struct {
	RatesUS    map[string]float64  `json:"rates_us"`
	MaxWork    map[string]int64    `json:"max_work"`
	Shape      RestoreServiceShape `json:"shape"`
	Provenance string              `json:"provenance"`
	Coverage   string              `json:"coverage"`
}

type restoreServiceCost struct{ config RestoreServiceCostConfig }

func serviceShape(c Config) RestoreServiceShape {
	cap := int64(0)
	if c.DecisionPolicy != nil {
		cap = c.DecisionPolicy.PrefillTokenCap
	}
	window := int64(1)
	if c.Mechanisms != nil {
		window = int64(max(1, c.Mechanisms.RestoreWindow))
	}
	return RestoreServiceShape{BlockTokens: c.BlockTokens, MaxBatchTokens: c.MaxBatchTokens, MaxSequences: c.MaxSequences, PrefillChunk: c.PrefillChunk, PrefillTokenCap: cap, RestoreWindowBlocks: window}
}

func newRestoreServiceCost(c Config) (*restoreServiceCost, error) {
	if c.DecisionPolicy == nil || c.DecisionPolicy.RestoreServiceCost == nil || c.DecisionEstimates == nil {
		return nil, fmt.Errorf("restore service cost requires restore estimates and a decision policy")
	}
	p := *c.DecisionPolicy.RestoreServiceCost
	want := serviceShape(c)
	want.MaxInputTokens = p.Shape.MaxInputTokens
	if p.Shape != want || want.MaxInputTokens <= 0 || strings.TrimSpace(p.Provenance) == "" || strings.TrimSpace(p.Coverage) == "" {
		return nil, fmt.Errorf("restore service profile shape/provenance/coverage mismatch")
	}
	if len(p.RatesUS) != len(restoreWorkNames) || len(p.MaxWork) != len(restoreWorkNames) {
		return nil, fmt.Errorf("restore service profile requires all known work counters")
	}
	rates, bounds := map[string]float64{}, map[string]int64{}
	for _, name := range restoreWorkNames {
		rate, ok := p.RatesUS[name]
		bound, known := p.MaxWork[name]
		if !ok || !known || math.IsNaN(rate) || math.IsInf(rate, 0) || rate < 0 || bound < 0 {
			return nil, fmt.Errorf("invalid restore service coefficient/bound %q", name)
		}
		rates[name], bounds[name] = rate, bound
	}
	p.RatesUS, p.MaxWork = rates, bounds
	return &restoreServiceCost{config: p}, nil
}

// RestoreDecisionWork counts operations performed by restoreEstimator using only
// visible state and its candidate endpoints. It never reads predicted completion
// times, final TTFT or future output lengths. No estimator/engine is re-executed.
func RestoreDecisionWork(c Config, v sim.DecisionView) (map[string]int64, error) {
	return restoreDecisionWork(serviceShape(c), v)
}

func restoreDecisionWork(shape RestoreServiceShape, v sim.DecisionView) (map[string]int64, error) {
	if v.Estimates == nil || !v.KV.TransfersKnown || v.BlockTokens <= 0 || v.MaxSequences <= 0 || shape.RestoreWindowBlocks <= 0 {
		return nil, fmt.Errorf("restore service work requires visible estimates and valid geometry")
	}
	w := map[string]int64{}
	for _, name := range restoreWorkNames {
		w[name] = 0
	}
	w["fixed"], w["visible_requests"], w["waiting_requests"], w["pending_transfers"] = 1, int64(len(v.Waiting)+len(v.Running)), int64(len(v.Waiting)), int64(len(v.KV.PendingTransfers))
	add := func(name string, count int64) error {
		if count < 0 || w[name] > math.MaxInt64-count {
			return fmt.Errorf("restore work counter overflow: %s", name)
		}
		w[name] += count
		return nil
	}
	requests := map[string]sim.DecisionRequest{}
	states := map[string]sim.DecisionKVRequest{}
	for _, r := range v.KV.Requests {
		states[r.ID] = r
	}
	actual := int64(0)
	for _, group := range [][]sim.DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			if r.InputTokens < 0 {
				return nil, fmt.Errorf("negative prompt length")
			}
			if err := add("prompt_blocks", r.InputTokens/v.BlockTokens); err != nil {
				return nil, err
			}
		}
	}
	for _, r := range v.Waiting {
		requests[r.ID] = r
	}
	for _, r := range v.Running {
		if r.ComputedTokens >= r.InputTokens {
			actual++
		}
	}
	for _, row := range v.Estimates.Requests {
		r, ok := requests[row.Request]
		state, known := states[row.Request]
		if !ok || !known {
			return nil, fmt.Errorf("missing request/prefix for service work")
		}
		for _, choice := range row.Choices {
			end, local := choice.MaxPrefixBlocks, state.LocalPrefixBlocks
			if end < local || end < 0 || end > math.MaxInt64/v.BlockTokens {
				return nil, fmt.Errorf("invalid restore endpoint in service work")
			}
			prefix := state.LocalPrefixTokens
			if end > local {
				prefix = min(end*v.BlockTokens, r.InputTokens-1)
			}
			for name, count := range map[string]int64{"candidates": 1, "load_windows": (end - local) / shape.RestoreWindowBlocks} {
				if err := add(name, count); err != nil {
					return nil, err
				}
			}
			if (end-local)%shape.RestoreWindowBlocks != 0 {
				if err := add("load_windows", 1); err != nil {
					return nil, err
				}
			}
			for _, decodes := range []int64{actual, v.MaxSequences - 1} {
				chunk := v.MaxBatchTokens - decodes
				for _, cap := range []int64{v.PrefillChunk, shape.PrefillTokenCap} {
					if cap > 0 {
						chunk = min(chunk, cap)
					}
				}
				if chunk <= 0 || prefix < 0 {
					return nil, fmt.Errorf("invalid compute chunk/prefix in service work")
				}
				remaining := max(int64(0), r.InputTokens-prefix)
				steps := remaining / chunk
				if remaining%chunk != 0 {
					steps++
				}
				if steps > math.MaxInt64/(1+decodes) {
					return nil, fmt.Errorf("compute item counter overflow")
				}
				if err := add("compute_steps", steps); err != nil {
					return nil, err
				}
				if err := add("compute_items", steps*(1+decodes)); err != nil {
					return nil, err
				}
			}
		}
	}
	return w, nil
}

func (m *restoreServiceCost) Estimate(v sim.DecisionView, _ sim.DecisionPlan) (sim.DecisionCostEstimate, error) {
	p := m.config
	shape := p.Shape
	if v.BlockTokens != shape.BlockTokens || v.MaxBatchTokens != shape.MaxBatchTokens || v.MaxSequences != shape.MaxSequences || v.PrefillChunk != shape.PrefillChunk {
		return sim.DecisionCostEstimate{}, fmt.Errorf("decision view outside service profile geometry")
	}
	for _, group := range [][]sim.DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			if r.InputTokens > shape.MaxInputTokens {
				return sim.DecisionCostEstimate{}, fmt.Errorf("request outside service profile prompt range")
			}
		}
	}
	w, err := restoreDecisionWork(shape, v)
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	total := 0.0
	for _, name := range restoreWorkNames {
		if w[name] > p.MaxWork[name] {
			return sim.DecisionCostEstimate{}, fmt.Errorf("decision work outside service profile: %s", name)
		}
		total += float64(w[name]) * p.RatesUS[name]
	}
	if math.IsInf(total, 0) || math.IsNaN(total) || total >= float64(math.MaxInt64) {
		return sim.DecisionCostEstimate{}, fmt.Errorf("decision service cost overflow")
	}
	return sim.DecisionCostEstimate{ExtraUS: int64(math.Ceil(total)), Provenance: p.Provenance + "; " + p.Coverage}, nil
}
