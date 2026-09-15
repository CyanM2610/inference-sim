package sim

import (
	"fmt"
	"math"
)

// PreemptionStoragePolicy supplies a fixed save/recompute baseline for victims
// selected by another request policy. It never chooses or releases a victim.
// Explicit per-victim choices made by the base policy take precedence.
type PreemptionStoragePolicy struct {
	base      DecisionPolicy
	mode      string
	pool      string
	guardband float64
}

func NewPreemptionStoragePolicy(base DecisionPolicy, mode, pool string) (DecisionPolicy, error) {
	if base == nil || mode != "spill" && mode != "recompute" || mode == "spill" && pool == "" || mode == "recompute" && pool != "" {
		return nil, fmt.Errorf("preemption storage requires a base policy and spill+pool or recompute without a pool")
	}
	p := &PreemptionStoragePolicy{base: base, mode: mode, pool: pool}
	return withPreemptionStorageEvents(p), nil
}

// Budget selection uses the base policy's retained ledger after its current
// decision. Requeue never manufactures a fresh arrival or replenishes slack.
func NewBudgetPreemptionStoragePolicy(base DecisionPolicy, pool string, guardband float64) (DecisionPolicy, error) {
	if base == nil || pool == "" || guardband < 1 || math.IsNaN(guardband) || math.IsInf(guardband, 0) {
		return nil, fmt.Errorf("budget spill requires base policy, pool and finite guardband >= 1")
	}
	if _, ok := base.(interface{ RequestBudgets(int64) []RequestBudget }); !ok {
		return nil, fmt.Errorf("budget spill requires a retained request-budget provider")
	}
	return withPreemptionStorageEvents(&PreemptionStoragePolicy{base: base, mode: "budget", pool: pool, guardband: guardband}), nil
}

func withPreemptionStorageEvents(p *PreemptionStoragePolicy) DecisionPolicy {
	if listener, ok := p.base.(DecisionEventListener); ok {
		return &preemptionStorageEvents{PreemptionStoragePolicy: p, listener: listener}
	}
	return p
}

func (p *PreemptionStoragePolicy) Decide(v DecisionView) DecisionPlan {
	if !v.Capabilities.RequestSpill {
		panic("preemption storage policy requires request_spill capability")
	}
	if p.mode == "budget" {
		if v.Estimates == nil || v.Estimates.SpillCoverage == "" {
			panic("budget spill requires explicit spill estimates")
		}
		if err := validateSpillEstimates(v, *v.Estimates); err != nil {
			panic(err)
		}
	}
	plan := p.base.Decide(v)
	budgets := map[string]RequestBudget{}
	if p.mode == "budget" {
		for _, b := range p.RequestBudgets(v.NowUS) {
			budgets[b.Request] = b
		}
	}
	plan.PreemptionStorage = append([]DecisionPreemptionStorage(nil), plan.PreemptionStorage...)
	seen := map[string]bool{}
	for _, action := range plan.PreemptionStorage {
		seen[action.Request] = true
	}
	victims := append([]string(nil), plan.Preemptions...)
	if plan.CapacityVictims != nil {
		victims = append(victims, plan.CapacityVictims.Order...)
	}
	for _, id := range victims {
		if !seen[id] {
			action := DecisionPreemptionStorage{Request: id, Mode: p.mode, Pool: p.pool}
			if p.mode == "budget" {
				action = p.budgetSpillChoice(v, id, budgets[id])
			}
			plan.PreemptionStorage = append(plan.PreemptionStorage, action)
			seen[id] = true
		}
	}
	return plan
}

func (p *PreemptionStoragePolicy) budgetSpillChoice(v DecisionView, id string, budget RequestBudget) DecisionPreemptionStorage {
	action := DecisionPreemptionStorage{Request: id, Mode: "recompute"}
	if !budget.Known || budget.RemainingUS < 0 {
		return action
	}
	complete := int64(0)
	for _, q := range v.KV.Requests {
		if q.ID == id {
			for _, target := range q.SpillTargets {
				if target.Pool == p.pool {
					complete = target.CompleteBlocks
				}
			}
		}
	}
	if complete == 0 {
		return action
	}
	for _, row := range v.Estimates.Spills {
		if row.Request != id || row.Pool != p.pool || row.Unavailable != "" || !row.CapacityPossible {
			continue
		}
		cost := math.Ceil(float64(row.StoreUS) * p.guardband)
		if cost >= float64(math.MaxInt64/16) {
			panic("guarded spill prediction overflow")
		}
		if int64(cost) <= budget.RemainingUS {
			action.Mode, action.Pool = "spill", p.pool
		}
		break
	}
	return action
}

func (p *PreemptionStoragePolicy) RequestBudgets(now int64) []RequestBudget {
	if base, ok := p.base.(interface{ RequestBudgets(int64) []RequestBudget }); ok {
		return base.RequestBudgets(now)
	}
	return nil
}

func (p *PreemptionStoragePolicy) Observe(f DecisionFeedback) { p.base.Observe(f) }
func (p *PreemptionStoragePolicy) Reset() {
	if base, ok := p.base.(interface{ Reset() }); ok {
		base.Reset()
	}
}
func (p *PreemptionStoragePolicy) AdditionalCostCoverage() string {
	coverage := "preemption_storage_policy_not_independently_calibrated"
	if base, ok := p.base.(interface{ AdditionalCostCoverage() string }); ok {
		coverage += ";" + base.AdditionalCostCoverage()
	}
	return coverage
}

type preemptionStorageEvents struct {
	*PreemptionStoragePolicy
	listener DecisionEventListener
}

func (p *preemptionStorageEvents) OnEvent(e DecisionEvent) { p.listener.OnEvent(e) }
