package sim

import "sort"

// NewBudgetPressurePolicy combines retained initial budgets, restore decisions,
// and a conditional largest-budget victim order. HBM failure detection and
// actual physical reclamation belong to the runtime, not this policy.
func NewBudgetPressurePolicy(guardband float64, prefillCap int64) (*BudgetRestorePolicy, error) {
	p, err := NewInitialBudgetRestorePolicy(guardband, prefillCap)
	if err == nil {
		p.capacityPressure = true
	}
	return p, err
}

// The ablation keeps the same queue, restore, reservation and pressure detection
// mechanisms; only victim order is the running-batch tail order (FCFS).
func NewBudgetPressureFIFOPolicy(guardband float64, prefillCap int64) (*BudgetRestorePolicy, error) {
	p, err := NewBudgetPressurePolicy(guardband, prefillCap)
	if err == nil {
		p.capacityFIFO = true
	}
	return p, err
}

func (p *BudgetRestorePolicy) addCapacityVictims(v DecisionView, plan *DecisionPlan) {
	if !v.Capabilities.CapacityPreemption || !v.Capabilities.PrefillPreemption {
		panic("budget_pressure requires capacity and prefill preemption")
	}
	var candidates []RequestBudget
	for _, r := range v.Running {
		b := p.initialBudgets[r.ID]
		if r.PrefillPreemptible && b.Known {
			b.RemainingUS = remainingInitialBudget(b, v.NowUS)
			candidates = append(candidates, b)
		}
	}
	if p.capacityFIFO {
		for i, j := 0, len(candidates)-1; i < j; i, j = i+1, j-1 {
			candidates[i], candidates[j] = candidates[j], candidates[i]
		}
	} else {
		sort.Slice(candidates, func(i, j int) bool {
			a, b := candidates[i], candidates[j]
			if a.RemainingUS != b.RemainingUS {
				return a.RemainingUS > b.RemainingUS
			}
			if a.ArrivalUS != b.ArrivalUS {
				return a.ArrivalUS > b.ArrivalUS
			}
			return a.Request > b.Request
		})
	}
	plan.CapacityVictims = &DecisionCapacityVictims{Order: []string{}}
	for _, b := range candidates {
		plan.CapacityVictims.Order = append(plan.CapacityVictims.Order, b.Request)
	}
	for _, r := range v.Waiting {
		guard := DecisionCapacityAdmission{Request: r.ID, Victims: []string{}}
		budget := p.initialBudgets[r.ID]
		if budget.Known {
			remaining := remainingInitialBudget(budget, v.NowUS)
			for _, b := range candidates {
				if b.RemainingUS > remaining || b.RemainingUS == remaining && (b.ArrivalUS > r.ArrivalUS || b.ArrivalUS == r.ArrivalUS && b.Request > r.ID) {
					guard.Victims = append(guard.Victims, b.Request)
				}
			}
		}
		plan.CapacityVictims.Admission = append(plan.CapacityVictims.Admission, guard)
	}
}
