package sim

import (
	"math"
	"sort"
)

// RequestBudget is a detached diagnostic, not an SLO guarantee. InitializedUS
// is the first usable decision-boundary estimate, which can follow arrival.
// Unknown budgets never stand in for a running request's missing history.
type RequestBudget struct {
	ReestimateCount        int64  `json:"reestimate_count,omitempty"`
	ReestimatedUS          int64  `json:"reestimated_us,omitempty"`
	ReestimatedIntrinsicUS int64  `json:"reestimated_intrinsic_us,omitempty"`
	ReestimatedBudgetUS    int64  `json:"reestimated_budget_us,omitempty"`
	ReestimateProvenance   string `json:"reestimate_provenance,omitempty"`
	ReestimateCoverage     string `json:"reestimate_coverage,omitempty"`
	ReestimatePending      bool   `json:"reestimate_pending,omitempty"`
	reestimateDecodeLimit  int64
	Request                string `json:"request"`
	Known                  bool   `json:"known"`
	Reason                 string `json:"reason,omitempty"`
	ArrivalUS              int64  `json:"arrival_us"`
	InputTokens            int64  `json:"input_tokens"`
	TTFTTargetUS           int64  `json:"ttft_target_us"`
	InitializedUS          int64  `json:"initialized_us"`
	IntrinsicUS            int64  `json:"intrinsic_us"`
	InitialUS              int64  `json:"initial_us"`
	RemainingUS            int64  `json:"remaining_us"`
	Provenance             string `json:"provenance,omitempty"`
	Coverage               string `json:"coverage,omitempty"`
}

// NewInitialBudgetRestorePolicy retains the first usable intrinsic estimate
// through execution and requeue. It adds neither dual queues nor preemption;
// those mechanisms must consume the same budget without manufacturing history.
func NewInitialBudgetRestorePolicy(guardband float64, prefillCap int64) (*BudgetRestorePolicy, error) {
	p, err := NewBudgetRestorePolicy(guardband, prefillCap)
	if err == nil {
		p.initialBudgets = map[string]RequestBudget{}
	}
	return p, err
}

// Reset begins a new workload with no retained requests or decision clock.
// A nil ledger remains nil so non-retained policies keep their existing mode.
func (p *BudgetRestorePolicy) Reset() {
	if p.initialBudgets != nil {
		p.initialBudgets = map[string]RequestBudget{}
	}
	p.budgetNowUS = 0
	if p.budgetReestimate != nil {
		p.budgetReestimate = &budgetReestimateState{}
	}
}

// AdditionalCostCoverage keeps new policy work visible even if the caller
// reuses a calibrated estimator or selects an existing decision cost model.
func (p *BudgetRestorePolicy) AdditionalCostCoverage() string {
	if p.capacityPressure {
		return "initial_budget_and_capacity_preemption_not_independently_calibrated"
	}
	if p.initialBudgets != nil {
		return "initial_budget_history_not_independently_calibrated"
	}
	return ""
}

func remainingInitialBudget(b RequestBudget, now int64) int64 {
	elapsed := max(int64(0), now-b.ArrivalUS)
	allocated := b.InitialUS
	if b.ReestimateCount > 0 {
		allocated = b.ReestimatedBudgetUS
	}
	if allocated < math.MinInt64+elapsed {
		return math.MinInt64
	}
	return allocated - elapsed
}

// RequestBudgets returns independent values in ID order, including unknown
// visible requests. No costs are recalculated by this diagnostic read.
func (p *BudgetRestorePolicy) RequestBudgets(now int64) []RequestBudget {
	if p.initialBudgets == nil {
		return nil
	}
	if now < p.budgetNowUS {
		panic("initial budget diagnostic time precedes last decision")
	}
	var rows []RequestBudget
	for _, b := range p.initialBudgets {
		if b.Known {
			b.RemainingUS = remainingInitialBudget(b, now)
		}
		rows = append(rows, b)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Request < rows[j].Request })
	return rows
}

func (p *BudgetRestorePolicy) updateInitialBudgets(v DecisionView, estimates map[string]DecisionRequestEstimate) {
	if p.initialBudgets == nil {
		return
	}
	if v.NowUS < p.budgetNowUS {
		panic("initial budget decision time moved backwards")
	}
	// Stage changes so malformed identities/estimates cannot partially reset
	// policy history. New maps also discard requests absent from the live view.
	next := map[string]RequestBudget{}
	visit := func(r DecisionRequest, waiting bool) {
		if r.ArrivalUS < 0 || r.ArrivalUS > v.NowUS || r.InputTokens <= 0 || r.TTFTTargetUS < 0 || r.ID == "" {
			panic("invalid request in initial budget view")
		}
		if _, duplicate := next[r.ID]; duplicate {
			panic("duplicate request in initial budget view")
		}
		b, seen := p.initialBudgets[r.ID]
		if seen && (b.ArrivalUS != r.ArrivalUS || b.InputTokens != r.InputTokens || b.TTFTTargetUS != r.TTFTTargetUS) {
			panic("initial budget request identity changed")
		}
		if !seen {
			b = RequestBudget{Request: r.ID, ArrivalUS: r.ArrivalUS, InputTokens: r.InputTokens, TTFTTargetUS: r.TTFTTargetUS}
		}
		if !b.Known {
			b.Reason = "missing_initial_estimate"
			if r.TTFTTargetUS == 0 {
				b.Reason = "no_ttft_target"
			} else if waiting && r.ComputedTokens < r.InputTokens {
				e := estimates[r.ID]
				if e.Unavailable != "" {
					b.Reason = e.Unavailable
				} else if len(e.Choices) > 0 {
					b.Known, b.Reason = true, ""
					b.IntrinsicUS = p.guardedCompute(e.Choices[len(e.Choices)-1])
					b.InitialUS = r.TTFTTargetUS - b.IntrinsicUS
					b.InitializedUS = v.NowUS
					b.Provenance, b.Coverage = v.Estimates.Provenance, v.Estimates.Coverage
				}
			}
		}
		next[r.ID] = b
	}
	for _, r := range v.Waiting {
		visit(r, true)
	}
	for _, r := range v.Running {
		visit(r, false)
	}
	if p.budgetReestimate != nil {
		p.budgetReestimate = p.reestimateBudgets(v, next, estimates)
	}
	p.initialBudgets, p.budgetNowUS = next, v.NowUS
}
