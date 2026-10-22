package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// WaitExecutionCostConfig prices only the observed native schedule-dispatch
// wrapper: applied, allocator attempts, failed allocators, actual preemptions,
// restore matches, token-reserve visits. ActionUS is the disjoint preempt/free
// interval. Decision, completion, idle/driver and registration stay separate.
type WaitExecutionCostConfig struct {
	Family          string              `json:"family"`
	RatesUS         []int64             `json:"rates_us"`
	ActionUS        int64               `json:"action_us"`
	MaxCounts       []int64             `json:"max_counts"`
	MaxInputTokens  int64               `json:"max_input_tokens"`
	MaxOutputTokens int                 `json:"max_output_tokens"`
	HBMBlocks       int64               `json:"hbm_blocks"`
	Shape           RestoreServiceShape `json:"shape"`
	Provenance      string              `json:"provenance"`
}

type waitExecutionCost struct{ config WaitExecutionCostConfig }

func newWaitExecutionCost(c Config) (*waitExecutionCost, error) {
	if c.DecisionPolicy == nil || c.DecisionPolicy.WaitExecutionCost == nil || c.DecisionPolicy.ExecutionCost != nil {
		return nil, fmt.Errorf("wait execution profile requires resident waits, control steps, one restore engine-phase instance and no other execution profile")
	}
	p := *c.DecisionPolicy.WaitExecutionCost
	if err := validateWaitServiceScope(c, p.Family, p.Shape, p.HBMBlocks, p.MaxInputTokens, p.MaxOutputTokens, p.Provenance); err != nil {
		return nil, err
	}
	if len(p.RatesUS) != 6 || len(p.MaxCounts) != 5 || p.ActionUS < 0 {
		return nil, fmt.Errorf("invalid wait execution family, geometry, coefficients or coverage")
	}
	for _, v := range append(append([]int64(nil), p.RatesUS...), p.MaxCounts...) {
		if v < 0 {
			return nil, fmt.Errorf("negative wait execution coefficient or bound")
		}
	}
	if p.Family == "basic" && (p.ActionUS != 0 || p.MaxCounts[1] != 0 || p.MaxCounts[2] != 0) {
		return nil, fmt.Errorf("basic wait profile has no measured failed allocator or preemption coverage")
	}
	p.RatesUS = append([]int64(nil), p.RatesUS...)
	p.MaxCounts = append([]int64(nil), p.MaxCounts...)
	return &waitExecutionCost{p}, nil
}

// WaitExecutionCounts returns native-equivalent counts while keeping the raw
// simulator calls intact. A failed async LOAD skips native token-reserve, but
// still visits the allocator. HIT_PENDING skips both native operations.
func WaitExecutionCounts(w sim.BatchExecutionWork, preemptions int64) ([]int64, error) {
	if !w.TokenChecksKnown || preemptions < 0 {
		return nil, fmt.Errorf("wait execution requires observed token checks")
	}
	counts := []int64{1, int64(len(w.Allocations)), 0, preemptions, 0, int64(len(w.TokenChecks))}
	if len(w.TokenChecks) == 0 && len(w.Allocations) > 0 {
		return nil, fmt.Errorf("allocator calls have no token check")
	}
	for i, check := range w.TokenChecks {
		if check.Phase != "running" && check.Phase != "waiting" {
			return nil, fmt.Errorf("unsupported token-check phase %q", check.Phase)
		}
		end := len(w.Allocations)
		if i+1 < len(w.TokenChecks) {
			end = w.TokenChecks[i+1].AllocationStart
		}
		if check.AllocationStart < 0 || end < check.AllocationStart || end > len(w.Allocations) || i == 0 && check.AllocationStart != 0 {
			return nil, fmt.Errorf("invalid token-check allocation range")
		}
		calls := w.Allocations[check.AllocationStart:end]
		if check.Held && (check.Phase != "running" || check.Tokens != 0 || len(calls) > 0) {
			return nil, fmt.Errorf("held check issued work")
		}
		reserve := true
		for _, a := range calls {
			if a.Request != check.Request {
				return nil, fmt.Errorf("allocator/check identity mismatch")
			}
			if a.Granted {
				if a.Failure != nil {
					return nil, fmt.Errorf("stale allocation failure")
				}
				continue
			}
			f := a.Failure
			if f == nil || f.Request != a.Request || f.RestoreCandidateBlocks < 0 {
				return nil, fmt.Errorf("unknown wait allocation outcome")
			}
			switch {
			case f.Kind == "capacity":
				counts[2]++
				if check.Phase == "waiting" {
					if !f.RestoreLookupKnown {
						return nil, fmt.Errorf("failed waiting allocation lacks observed restore lookup")
					}
					if f.RestoreCandidateBlocks > 0 {
						reserve = false
					}
				}
			case f.Kind == "wait" && f.Reason == "restore_submitted":
				if check.Phase != "waiting" {
					return nil, fmt.Errorf("LOAD submitted outside waiting admission")
				}
				reserve = false
			case f.Kind == "wait" && f.Reason == "restore_lookup_pending":
				if check.Phase != "waiting" || len(calls) != 1 {
					return nil, fmt.Errorf("invalid deferred lookup")
				}
				reserve = false
				counts[1]--
			default:
				return nil, fmt.Errorf("wait execution profile does not cover %s/%s", f.Kind, f.Reason)
			}
		}
		if !reserve {
			counts[5]--
		}
	}
	for _, query := range w.Prefixes {
		if !query.AdoptingRestore {
			counts[4]++
		}
	}
	return counts, nil
}

func (m *waitExecutionCost) EstimateExecution(w sim.BatchExecutionWork, preemptions int64) (sim.DecisionCostEstimate, error) {
	counts, err := WaitExecutionCounts(w, preemptions)
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	var total int64
	for i, n := range counts {
		if n < 0 || i > 0 && n > m.config.MaxCounts[i-1] {
			return sim.DecisionCostEstimate{}, fmt.Errorf("wait execution feature %d exceeds profile coverage: %d", i, n)
		}
		rate := m.config.RatesUS[i]
		if n > 0 && rate > (math.MaxInt64-total)/n {
			return sim.DecisionCostEstimate{}, fmt.Errorf("wait execution fee overflow")
		}
		total += rate * n
	}
	if preemptions > 0 && m.config.ActionUS > (math.MaxInt64-total)/preemptions {
		return sim.DecisionCostEstimate{}, fmt.Errorf("wait action fee overflow")
	}
	total += m.config.ActionUS * preemptions
	return sim.DecisionCostEstimate{ExtraUS: total, Provenance: m.config.Provenance}, nil
}
