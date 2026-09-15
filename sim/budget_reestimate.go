package sim

import "math"

// budgetReestimateState observes current decode occupancy, not output lengths
// or future completions. A request's pending opportunity lives with its ledger.
type budgetReestimateState struct {
	seen        bool
	decodeCount int64
}

func (p *BudgetRestorePolicy) reestimateBudgets(v DecisionView, next map[string]RequestBudget, estimates map[string]DecisionRequestEstimate) *budgetReestimateState {
	var decodes int64
	for _, r := range v.Running {
		if r.EmittedTokens > 0 {
			decodes++
		}
	}
	dropped := p.budgetReestimate.seen && decodes < p.budgetReestimate.decodeCount
	kv := map[string]DecisionKVRequest{}
	for _, state := range v.KV.Requests {
		kv[state.ID] = state
	}
	visit := func(r DecisionRequest, waiting bool) {
		b := next[r.ID]
		clear := func() { b.ReestimatePending, b.reestimateDecodeLimit = false, 0 }
		if !b.Known || r.TTFTTargetUS == 0 || r.EmittedTokens > 0 || r.ComputedTokens >= r.InputTokens {
			clear()
			next[r.ID] = b
			return
		}
		if b.ReestimatePending && decodes > b.reestimateDecodeLimit {
			clear() // A rebound invalidates an unconsumed load-drop opportunity.
		}
		if dropped && p.initialBudgets[r.ID].Known && remainingInitialBudget(b, v.NowUS) <= 0 {
			b.ReestimatePending, b.reestimateDecodeLimit = true, decodes
		}
		state, known := kv[r.ID]
		e := estimates[r.ID]
		if waiting && b.ReestimatePending && known && !state.TransferPending && !state.Deferred && e.Unavailable == "" && len(e.Choices) > 0 {
			if b.ReestimateCount == math.MaxInt64 {
				panic("budget reestimate counter overflow")
			}
			b.ReestimatedIntrinsicUS = p.guardedCompute(e.Choices[len(e.Choices)-1])
			b.ReestimatedBudgetUS = b.TTFTTargetUS - b.ReestimatedIntrinsicUS
			b.ReestimatedUS, b.ReestimateCount = v.NowUS, b.ReestimateCount+1
			b.ReestimateProvenance, b.ReestimateCoverage = v.Estimates.Provenance, v.Estimates.Coverage
			clear()
		}
		next[r.ID] = b
	}
	for _, r := range v.Waiting {
		visit(r, true)
	}
	for _, r := range v.Running {
		visit(r, false)
	}
	// Commit the load clock only after every staged estimate has succeeded.
	return &budgetReestimateState{seen: true, decodeCount: decodes}
}
