package sim

import (
	"fmt"
	"math"
	"sort"
)

// BudgetRestorePolicy combines least-remaining-budget queue order with
// budget-feasible loading. It is a Cascade-inspired subset, not its dual queues
// or preemption policy. Estimates are inputs, never execution guarantees.
type BudgetRestorePolicy struct {
	budgetReestimate *budgetReestimateState
	capacityPressure bool
	capacityFIFO     bool
	initialBudgets   map[string]RequestBudget
	budgetNowUS      int64
	completionBound  bool
	guardband        float64
	prefillCap       int64
}

// NewCompletionBudgetRestorePolicy charges the remaining computation of each
// candidate, not the full-hit computation of a different candidate. When no
// candidate meets the predicted deadline it chooses minimum predicted service
// as best effort; it does not turn an infeasible request into a claimed success.
func NewCompletionBudgetRestorePolicy(guardband float64, prefillCap int64) (*BudgetRestorePolicy, error) {
	p, err := NewBudgetRestorePolicy(guardband, prefillCap)
	if err == nil {
		p.completionBound = true
	}
	return p, err
}

func (p *BudgetRestorePolicy) guardedCompute(c DecisionRestoreEstimate) int64 {
	x := math.Ceil(p.guardband * (float64(c.ComputeUS) + float64(c.MaxLoadComputeUS)) / 2)
	if x >= float64(math.MaxInt64/4) {
		panic("budget restore service estimate overflows")
	}
	return int64(x)
}

func NewBudgetRestorePolicy(guardband float64, prefillCap int64) (*BudgetRestorePolicy, error) {
	if math.IsNaN(guardband) || math.IsInf(guardband, 0) || guardband < 1 || prefillCap < 0 {
		return nil, fmt.Errorf("budget restore requires finite guardband >= 1 and nonnegative prefill cap")
	}
	return &BudgetRestorePolicy{guardband: guardband, prefillCap: prefillCap}, nil
}

func (p *BudgetRestorePolicy) Decide(v DecisionView) DecisionPlan {
	if !v.Capabilities.RestoreChoice || !v.Capabilities.CostEstimates || v.Estimates == nil {
		panic("budget restore requires restore actions and explicit cost estimates")
	}
	byID := map[string]DecisionRequestEstimate{}
	for _, r := range v.Estimates.Requests {
		byID[r.Request] = r
	}
	p.updateInitialBudgets(v, byID)
	kv := map[string]DecisionKVRequest{}
	for _, r := range v.KV.Requests {
		kv[r.ID] = r
	}
	budget := map[string]int64{}
	plan := DecisionPlan{Version: v.Version}
	for _, r := range v.Waiting {
		budget[r.ID] = math.MaxInt64
		if b, ok := p.initialBudgets[r.ID]; ok && b.Known {
			budget[r.ID] = remainingInitialBudget(b, v.NowUS)
		}
		if r.ComputedTokens >= r.InputTokens {
			continue
		}
		e := byID[r.ID]
		if e.Unavailable != "" {
			if e.Unavailable == "decode_recompute_not_profiled" && r.EmittedTokens > 0 && r.RecomputeUntilTokens > r.InputTokens {
				// This TTFT policy has already observed the first output. Leave
				// history restoration to the demand path instead of inventing a
				// new first-token budget from an unprofiled replay prediction.
				continue
			}
			if !kv[r.ID].TransferPending && e.Unavailable != "no_free_sequence_slot" {
				panic("budget restore lacks a usable prediction: " + e.Unavailable)
			}
			continue // never rewrite an in-flight load
		}
		if len(e.Choices) == 0 {
			panic("budget restore missing request estimates")
		}
		local := e.Choices[0]
		full := e.Choices[len(e.Choices)-1]
		intrinsic := p.guardedCompute(full)
		if r.TTFTTargetUS > 0 && p.initialBudgets == nil {
			elapsed := max(int64(0), v.NowUS-r.ArrivalUS)
			// Saturate only the priority/budget diagnostic, never simulation time.
			remaining := r.TTFTTargetUS - elapsed
			if remaining < math.MinInt64+int64(intrinsic) {
				remaining = math.MinInt64
			} else {
				remaining -= int64(intrinsic)
			}
			budget[r.ID] = remaining
		}
		chosen := local
		for _, c := range e.Choices[1:] {
			if r.TTFTTargetUS > 0 {
				if c.LoadUS <= budget[r.ID] && c.LoadUS+c.ComputeUS <= local.ComputeUS {
					chosen = c
				}
			} else if c.LoadUS+c.ComputeUS < chosen.LoadUS+chosen.ComputeUS {
				chosen = c // no SLO: explicit minimum predicted service fallback
			}
		}
		if p.completionBound && r.TTFTTargetUS > 0 {
			remaining := r.TTFTTargetUS - max(int64(0), v.NowUS-r.ArrivalUS)
			feasible := false
			fastest := local
			for _, c := range e.Choices {
				if c.LoadUS+c.ComputeUS < fastest.LoadUS+fastest.ComputeUS {
					fastest = c
				}
				if c.LoadUS+p.guardedCompute(c) <= remaining && c.LoadUS+c.ComputeUS <= local.ComputeUS {
					chosen = c
					feasible = true
				}
			}
			if !feasible {
				chosen = fastest
			}
		}
		plan.Restores = append(plan.Restores, DecisionRestore{Request: r.ID, MaxPrefixBlocks: chosen.MaxPrefixBlocks})
	}
	requests := append([]DecisionRequest(nil), v.Waiting...)
	sort.SliceStable(requests, func(i, j int) bool {
		a, b := requests[i], requests[j]
		if budget[a.ID] != budget[b.ID] {
			return budget[a.ID] < budget[b.ID]
		}
		if a.ArrivalUS != b.ArrivalUS {
			return a.ArrivalUS < b.ArrivalUS
		}
		return a.ID < b.ID
	})
	for _, r := range requests {
		plan.QueueOrder = append(plan.QueueOrder, r.ID)
	}
	if p.prefillCap > 0 {
		for _, r := range append(requests, v.Running...) {
			if r.ComputedTokens < r.InputTokens {
				plan.TokenCaps = append(plan.TokenCaps, DecisionTokenCap{Request: r.ID, Tokens: p.prefillCap})
			}
		}
	}
	if p.capacityPressure {
		p.addCapacityVictims(v, &plan)
	}
	return plan
}

func (*BudgetRestorePolicy) Observe(DecisionFeedback) {}
