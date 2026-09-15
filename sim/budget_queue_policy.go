package sim

import (
	"fmt"
	"math"
	"sort"
)

// BudgetQueuePolicy adds two prefill queues and work-conserving token shares to
// the retained-budget/capacity policy. Shares are targets under contention, not
// promises that a blocked request can acquire a sequence slot or physical KV.
// No resource state, transfer completion, or future output is owned here.
type BudgetQueuePolicy struct {
	base                      *BudgetRestorePolicy
	primaryWeight, bestWeight int64
	pending                   *budgetQueueDispatch
	service                   BudgetQueueService
}

// BudgetQueueService is detached accounting of actual grants. DeficitUnits is
// bestWeight * contended prefill work - WeightSum * best-effort work in the
// current contention epoch. Positive values favor best effort in later plans.
type BudgetQueueService struct {
	PrimaryTokens            int64 `json:"primary_tokens"`
	BestEffortTokens         int64 `json:"best_effort_tokens"`
	ContendedPrimaryTokens   int64 `json:"contended_primary_tokens"`
	ContendedBestTokens      int64 `json:"contended_best_effort_tokens"`
	BestEffortUnservedRounds int64 `json:"best_effort_unserved_rounds"`
	AppliedDecisions         int64 `json:"applied_decisions"`
	DeficitUnits             int64 `json:"deficit_units"`
	WeightSum                int64 `json:"weight_sum"`
}

type budgetQueueItem struct {
	r       DecisionRequest
	demand  int64
	running bool
	prefill bool
	best    bool
	ready   bool
}

type budgetQueueDispatch struct {
	version          uint64
	limit            int64
	items            map[string]budgetQueueItem
	contended        bool
	reservationAware bool
}

type BudgetQueueOptions struct {
	ReestimateOnDecodeDrop bool
}

// NewBudgetQueuePolicy retains the original fixed-budget behavior. Positive
// integer weights, e.g. 4:1, set a best-effort token share target, not a full
// Cascade guarantee under arbitrary resource pressure.
func NewBudgetQueuePolicy(guardband float64, prefillCap, primaryWeight, bestWeight int64) (*BudgetQueuePolicy, error) {
	return NewBudgetQueuePolicyWithOptions(guardband, prefillCap, primaryWeight, bestWeight, BudgetQueueOptions{})
}

func NewBudgetQueuePolicyWithOptions(guardband float64, prefillCap, primaryWeight, bestWeight int64, options BudgetQueueOptions) (*BudgetQueuePolicy, error) {
	if primaryWeight <= 0 || bestWeight <= 0 || primaryWeight > 1000000-bestWeight {
		return nil, fmt.Errorf("budget queues require positive weights with sum <= 1000000")
	}
	base, err := NewBudgetPressurePolicy(guardband, prefillCap)
	if err != nil {
		return nil, err
	}
	if options.ReestimateOnDecodeDrop {
		base.budgetReestimate = &budgetReestimateState{}
	}
	p := &BudgetQueuePolicy{base: base, primaryWeight: primaryWeight, bestWeight: bestWeight}
	p.service.WeightSum = primaryWeight + bestWeight
	return p, nil
}

func (p *BudgetQueuePolicy) AdditionalCostCoverage() string {
	if p.base.budgetReestimate != nil {
		return "budget_queue_reestimate_not_independently_calibrated"
	}
	return "budget_queues_actual_grant_accounting_not_independently_calibrated"
}

func (p *BudgetQueuePolicy) ServiceState() BudgetQueueService { return p.service }
func (p *BudgetQueuePolicy) RequestBudgets(now int64) []RequestBudget {
	return p.base.RequestBudgets(now)
}
func (p *BudgetQueuePolicy) Reset() {
	p.base.Reset()
	p.pending = nil
	p.service = BudgetQueueService{WeightSum: p.primaryWeight + p.bestWeight}
}

func queueAdd(a, b int64) int64 {
	if b > 0 && a > math.MaxInt64-b || b < 0 && a < math.MinInt64-b {
		panic("budget queue service accounting overflow")
	}
	return a + b
}

func queueProduct(a, b int64) int64 {
	if a < 0 || b < 0 || b > 0 && a > math.MaxInt64/b {
		panic("budget queue weight overflow")
	}
	return a * b
}

func (p *BudgetQueuePolicy) Decide(v DecisionView) DecisionPlan {
	if p.pending != nil {
		panic("budget queue decision requires feedback for the previous version")
	}
	if !v.Capabilities.AdmissionSelection || !v.Capabilities.TokenCaps || !v.Capabilities.QueueOrder ||
		!v.KV.PrefixStateKnown || !v.KV.TransfersKnown || v.MaxBatchTokens <= 0 ||
		v.MaxBatchTokens > math.MaxInt64/p.service.WeightSum || v.MaxSequences <= 0 {
		panic("budget queues require admission, quotas, known KV state and bounded batch limits")
	}
	plan := p.base.Decide(v)
	states := map[string]DecisionKVRequest{}
	for _, state := range v.KV.Requests {
		states[state.ID] = state
	}
	items := map[string]budgetQueueItem{}
	var running, primary, best, protected []budgetQueueItem
	var readyPrimary, readyBest bool
	visit := func(r DecisionRequest, isRunning bool) {
		state, known := states[r.ID]
		if !known {
			panic("budget queue request lacks a KV state")
		}
		prefill := r.EmittedTokens == 0 && r.ComputedTokens < r.InputTokens
		b := p.base.initialBudgets[r.ID]
		be := prefill && (!b.Known || remainingInitialBudget(b, v.NowUS) <= 0)
		remaining := r.InputTokens - r.ComputedTokens
		if !prefill {
			remaining = max(int64(1), r.RecomputeUntilTokens-r.ComputedTokens)
		}
		demand := min(v.MaxBatchTokens, max(int64(1), remaining))
		if remaining > 1 && v.PrefillChunk > 0 {
			demand = min(demand, v.PrefillChunk)
		}
		if prefill && p.base.prefillCap > 0 {
			demand = min(demand, p.base.prefillCap)
		}
		ready := r.WaitUntilUS <= v.NowUS && !state.TransferPending && !state.Deferred
		if v.Capabilities.CapacityReservations {
			c := state.CapacityReservation
			if c == nil || c.NeededBlocks < 0 || c.FreeBlocks < 0 || c.Fits != (c.NeededBlocks <= c.FreeBlocks) {
				panic("budget queue request lacks a valid capacity reservation view")
			}
			canReclaim := false
			if plan.CapacityVictims != nil {
				canReclaim = len(plan.CapacityVictims.Order) > 0
				for _, a := range plan.CapacityVictims.Admission {
					if a.Request == r.ID {
						canReclaim = len(a.Victims) > 0
						break
					}
				}
			}
			if !isRunning && !c.Fits && !canReclaim {
				ready = false
			}
		}
		item := budgetQueueItem{r: r, demand: demand, running: isRunning, prefill: prefill, best: be, ready: ready}
		items[r.ID] = item
		if prefill && ready {
			readyBest = readyBest || be
			readyPrimary = readyPrimary || !be
		}
		if isRunning {
			running = append(running, item)
		} else if ready {
			if !prefill {
				protected = append(protected, item)
			} else if be {
				best = append(best, item)
			} else {
				primary = append(primary, item)
			}
		}
	}
	for _, r := range v.Running {
		visit(r, true)
	}
	for _, r := range v.Waiting {
		visit(r, false)
	}
	contended := readyPrimary && readyBest
	debt := p.service.DeficitUnits
	if !contended {
		debt = 0
	}
	order := func(group []budgetQueueItem, lrbf bool) {
		sort.SliceStable(group, func(i, j int) bool {
			a, b := group[i].r, group[j].r
			if lrbf {
				x, y := remainingInitialBudget(p.base.initialBudgets[a.ID], v.NowUS), remainingInitialBudget(p.base.initialBudgets[b.ID], v.NowUS)
				if x != y {
					return x < y
				}
			}
			if a.ArrivalUS != b.ArrivalUS {
				return a.ArrivalUS < b.ArrivalUS
			}
			return a.ID < b.ID
		})
	}
	order(primary, true)
	order(best, false)
	order(protected, false)
	selected := append([]budgetQueueItem(nil), running...)
	var plannedPrimary, plannedBest int64
	for _, item := range running {
		if !item.prefill || !item.ready {
			continue
		}
		if item.best {
			plannedBest = queueAdd(plannedBest, item.demand)
		} else {
			plannedPrimary = queueAdd(plannedPrimary, item.demand)
		}
	}
	slots := max(int64(0), v.MaxSequences-int64(len(running)))
	for slots > 0 && (len(protected)+len(primary)+len(best) > 0) {
		var item budgetQueueItem
		switch {
		case len(protected) > 0:
			item, protected = protected[0], protected[1:]
		case len(best) > 0 && (len(primary) == 0 || queueAdd(queueProduct(p.bestWeight, queueAdd(queueAdd(plannedPrimary, plannedBest), 1)), debt) > queueProduct(p.service.WeightSum, plannedBest)):
			item, best = best[0], best[1:]
		case len(primary) > 0:
			item, primary = primary[0], primary[1:]
		default:
			item, best = best[0], best[1:]
		}
		selected = append(selected, item)
		if item.prefill {
			if item.best {
				plannedBest = queueAdd(plannedBest, item.demand)
			} else {
				plannedPrimary = queueAdd(plannedPrimary, item.demand)
			}
		}
		slots--
	}
	// Running requests require a positive grant unless the existing wait runtime
	// can suspend them. Decode/replay work gets room before either prefill queue.
	caps := map[string]int64{}
	var pmin, bmin, pmax, bmax int64
	var pg, bg, dg []budgetQueueItem
	for _, item := range selected {
		if !item.ready && item.prefill && v.Capabilities.PrefillWaits && item.r.PrefillWaitable && item.r.WaitUntilUS > v.NowUS {
			continue
		}
		if !item.prefill {
			dg = append(dg, item)
		} else if item.best {
			bg = append(bg, item)
			bmax = queueAdd(bmax, item.demand)
			if item.running {
				bmin++
			}
		} else {
			pg = append(pg, item)
			pmax = queueAdd(pmax, item.demand)
			if item.running {
				pmin++
			}
		}
	}
	budget := v.MaxBatchTokens
	for _, item := range dg {
		if item.running {
			caps[item.r.ID] = 1
			budget--
		}
	}
	if budget < pmin+bmin {
		panic("batch budget cannot cover running requests")
	}
	for _, item := range dg {
		add := min(item.demand-caps[item.r.ID], budget-pmin-bmin)
		caps[item.r.ID] += add
		budget -= add
	}
	total := min(budget, queueAdd(pmax, bmax))
	numerator := queueAdd(total*p.bestWeight, debt)
	wantBest := numerator / p.service.WeightSum
	if numerator > 0 && numerator%p.service.WeightSum != 0 {
		wantBest++
	}
	bestBudget := max(bmin, total-pmax, min(wantBest, bmax, total-pmin))
	primaryBudget := total - bestBudget
	allocate := func(group []budgetQueueItem, quota int64, lrbf bool) {
		for _, item := range group {
			if item.running {
				caps[item.r.ID] = 1
				quota--
			}
		}
		order(group, lrbf)
		for _, item := range group {
			add := min(item.demand-caps[item.r.ID], quota)
			caps[item.r.ID] += add
			quota -= add
		}
		if quota != 0 {
			panic("unassigned budget queue quota")
		}
	}
	allocate(pg, primaryBudget, true)
	allocate(bg, bestBudget, false)
	plan.TokenCaps = nil
	plan.QueueOrder = nil
	plan.Admission = &DecisionAdmission{Requests: []string{}}
	seen := map[string]bool{}
	for _, item := range selected {
		if caps[item.r.ID] > 0 {
			plan.TokenCaps = append(plan.TokenCaps, DecisionTokenCap{Request: item.r.ID, Tokens: caps[item.r.ID]})
			if !item.running {
				plan.Admission.Requests = append(plan.Admission.Requests, item.r.ID)
				plan.QueueOrder = append(plan.QueueOrder, item.r.ID)
				seen[item.r.ID] = true
			}
		}
	}
	// Keep a complete waiting permutation, including I/O/deadline-blocked rows.
	waiting := append([]DecisionRequest(nil), v.Waiting...)
	sort.SliceStable(waiting, func(i, j int) bool {
		a, b := items[waiting[i].ID], items[waiting[j].ID]
		if a.best != b.best {
			return !a.best
		}
		if !a.best && a.prefill && b.prefill {
			x, y := remainingInitialBudget(p.base.initialBudgets[a.r.ID], v.NowUS), remainingInitialBudget(p.base.initialBudgets[b.r.ID], v.NowUS)
			if x != y {
				return x < y
			}
		}
		if a.r.ArrivalUS != b.r.ArrivalUS {
			return a.r.ArrivalUS < b.r.ArrivalUS
		}
		return a.r.ID < b.r.ID
	})
	for _, r := range waiting {
		if !seen[r.ID] {
			plan.QueueOrder = append(plan.QueueOrder, r.ID)
		}
	}
	p.pending = &budgetQueueDispatch{version: v.Version, limit: v.MaxBatchTokens, items: items, contended: contended,
		reservationAware: v.Capabilities.CapacityReservations}
	return plan
}

func (p *BudgetQueuePolicy) Observe(f DecisionFeedback) {
	if p.pending == nil && f.Status == "rejected" && len(f.Grants) == 0 {
		return
	}
	if p.pending == nil || f.Version != p.pending.version {
		panic("budget queue feedback has no matching decision")
	}
	if f.Status != "applied" {
		if f.Status != "superseded" && f.Status != "rejected" {
			panic("unknown budget queue feedback status")
		}
		if len(f.Grants) != 0 {
			panic("unapplied budget queue feedback contains grants")
		}
		p.pending = nil
		return
	}
	var primary, best, total int64
	seen := map[string]bool{}
	for _, grant := range f.Grants {
		item, ok := p.pending.items[grant.Request]
		if !ok || seen[grant.Request] || grant.Tokens <= 0 {
			panic("invalid actual budget queue grant")
		}
		seen[grant.Request] = true
		total = queueAdd(total, grant.Tokens)
		if total > p.pending.limit {
			panic("actual budget queue grants exceed batch limit")
		}
		if !item.prefill {
			continue
		}
		if item.best {
			best = queueAdd(best, grant.Tokens)
		} else {
			primary = queueAdd(primary, grant.Tokens)
		}
	}
	contended := p.pending.contended
	if contended && p.pending.reservationAware {
		// Individually feasible new arrivals may still compete for the same
		// space in this step. Final failed admissions are not delivered service.
		blocked := map[string]bool{}
		for _, outcome := range f.Capacity {
			if _, ok := p.pending.items[outcome.Failure.Request]; !ok {
				panic("budget queue capacity feedback names an unobserved request")
			}
			if outcome.Failure.Kind == "capacity" && outcome.Status == "no_eligible_victim" {
				blocked[outcome.Failure.Request] = true
			}
		}
		var readyPrimary, readyBest bool
		for id, item := range p.pending.items {
			if !item.prefill || !item.ready || blocked[id] && !seen[id] {
				continue
			}
			readyBest = readyBest || item.best
			readyPrimary = readyPrimary || !item.best
		}
		contended = readyPrimary && readyBest
	}
	next := p.service
	next.PrimaryTokens = queueAdd(next.PrimaryTokens, primary)
	next.BestEffortTokens = queueAdd(next.BestEffortTokens, best)
	next.AppliedDecisions = queueAdd(next.AppliedDecisions, 1)
	if contended {
		next.ContendedPrimaryTokens = queueAdd(next.ContendedPrimaryTokens, primary)
		next.ContendedBestTokens = queueAdd(next.ContendedBestTokens, best)
		if primary > math.MaxInt64/p.bestWeight || best > math.MaxInt64/p.primaryWeight {
			panic("budget queue feedback overflow")
		}
		next.DeficitUnits = queueAdd(next.DeficitUnits, primary*p.bestWeight-best*p.primaryWeight)
		if best == 0 && primary > 0 {
			next.BestEffortUnservedRounds = queueAdd(next.BestEffortUnservedRounds, 1)
		}
	} else {
		next.DeficitUnits = 0 // idle/noncontended epochs do not create future credit
	}
	p.service, p.pending = next, nil
}
