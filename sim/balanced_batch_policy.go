package sim

import (
	"fmt"
	"math"
	"sort"
)

// BalancedBatchPolicy follows the head-first/deprioritize/refill structure of
// Strata's balanced batching. Ratios use currently recoverable load tokens and
// proposed query tokens. It is not layer-wise I/O, bundle hits or bubble filling.
// The runtime retains all allocation, transfer and completion responsibilities.
type BalancedBatchPolicy struct {
	readyComputeBudget  bool
	bundleHits          bool
	MaxLoadComputeRatio float64
	TokenBudget         int64
	MaxRequests         int64
	PrefillCap          int64
}

type BalancedBatchOptions struct {
	BundleHits         bool
	ReadyComputeBudget bool
}

func NewBalancedBatchPolicyWithOptions(ratio float64, budget, requests, cap int64, options BalancedBatchOptions) (*BalancedBatchPolicy, error) {
	p, err := NewBalancedBatchPolicy(ratio, budget, requests, cap)
	if err == nil {
		p.bundleHits, p.readyComputeBudget = options.BundleHits, options.ReadyComputeBudget
	}
	return p, err
}

func NewBalancedBatchPolicy(ratio float64, budget, requests, cap int64) (*BalancedBatchPolicy, error) {
	if math.IsNaN(ratio) || math.IsInf(ratio, 0) || ratio <= 0 || budget < 0 || requests < 0 || cap < 0 {
		return nil, fmt.Errorf("balanced batch requires a positive finite ratio and nonnegative limits")
	}
	return &BalancedBatchPolicy{MaxLoadComputeRatio: ratio, TokenBudget: budget, MaxRequests: requests, PrefillCap: cap}, nil
}

func NewBundleBalancedPolicy(ratio float64, budget, requests, cap int64) (*BalancedBatchPolicy, error) {
	return NewBalancedBatchPolicyWithOptions(ratio, budget, requests, cap, BalancedBatchOptions{BundleHits: true})
}

func (p *BalancedBatchPolicy) Decide(v DecisionView) DecisionPlan {
	if p.readyComputeBudget && (!v.Capabilities.BatchTokenCap || !v.Capabilities.RestoreDefersCompute) {
		panic("ready compute budget requires a runtime batch cap and deferred restore semantics")
	}
	if p.bundleHits && !v.KV.PrefixSharingKnown {
		panic("bundle balancing requires explicit prefix sharing state")
	}
	if !v.Capabilities.AdmissionSelection || !v.Capabilities.TokenCaps || !v.Capabilities.RestoreChoice || !v.KV.PrefixStateKnown || v.BlockTokens <= 0 {
		panic("balanced batch requires admission, token caps and recoverable prefix state")
	}
	budget, slots := v.MaxBatchTokens, v.MaxSequences
	if p.TokenBudget > 0 {
		budget = min(budget, p.TokenBudget)
	}
	if p.MaxRequests > 0 {
		slots = min(slots, p.MaxRequests)
	}
	plan := DecisionPlan{Version: v.Version, Admission: &DecisionAdmission{Requests: []string{}}}
	if p.readyComputeBudget {
		plan.BatchTokenCap = budget
	}
	var compute, load int64
	quota := func(remaining int64) int64 {
		q := min(max(int64(1), remaining), budget)
		for _, cap := range []int64{v.PrefillChunk, p.PrefillCap} {
			if cap > 0 {
				q = min(q, cap)
			}
		}
		return q
	}
	// Running requests keep their runtime order and are never implicitly evicted.
	// Positive quotas can shrink their prefill chunks, but cannot suspend decode.
	if budget < int64(len(v.Running)) {
		panic("balanced token budget cannot cover one token per running request")
	}
	for i, r := range v.Running {
		q := min(quota(r.InputTokens-r.ComputedTokens), budget-int64(len(v.Running)-i-1))
		plan.TokenCaps = append(plan.TokenCaps, DecisionTokenCap{Request: r.ID, Tokens: q})
		budget -= q
		compute += q
	}
	slots = max(int64(0), slots-int64(len(v.Running)))
	states := map[string]DecisionKVRequest{}
	for _, r := range v.KV.Requests {
		states[r.ID] = r
	}
	if p.bundleHits {
		for _, r := range v.Waiting {
			state, ok := states[r.ID]
			if !ok {
				panic("bundle batch missing request prefix state")
			}
			// A follower must query its final logits token. A restored full
			// final block is not a shareable hit for another fresh request.
			state.RecoverablePrefixBlocks = max(state.LocalPrefixBlocks, min(state.RecoverablePrefixBlocks, max(int64(0), r.InputTokens-1)/v.BlockTokens))
			states[r.ID] = state
		}
	}
	waiting := append([]DecisionRequest(nil), v.Waiting...)
	sort.SliceStable(waiting, func(i, j int) bool {
		if waiting[i].ArrivalUS != waiting[j].ArrivalUS {
			return waiting[i].ArrivalUS < waiting[j].ArrivalUS
		}
		return waiting[i].ID < waiting[j].ID
	})
	selected := map[string]bool{}
	plannedLoads := map[string]bool{}
	var deferred []DecisionRequest
	consider := func(r DecisionRequest, force bool) bool {
		if selected[r.ID] || slots <= 0 || budget <= 0 {
			return false
		}
		state, ok := states[r.ID]
		if !ok {
			panic("balanced batch missing request prefix state")
		}
		if state.TransferPending || r.WaitUntilUS > v.NowUS {
			return false
		}
		prefix := state.LocalPrefixTokens
		if state.RecoverablePrefixBlocks > state.LocalPrefixBlocks {
			prefix = min(state.RecoverablePrefixBlocks*v.BlockTokens, r.InputTokens-1)
		}
		q := quota(r.InputTokens - prefix)
		charged := q
		if p.readyComputeBudget && state.RecoverablePrefixBlocks > state.LocalPrefixBlocks {
			charged = 0
		}
		l := (state.RecoverablePrefixBlocks - state.LocalPrefixBlocks) * v.BlockTokens
		var keys []string
		if p.bundleHits {
			var usable bool
			keys, usable = decisionRestoreKeys(state)
			if !usable {
				return false
			}
			l = 0
			for _, key := range keys {
				if !plannedLoads[key] {
					l += v.BlockTokens
				}
			}
		}
		if !force && float64(load+l) > p.MaxLoadComputeRatio*float64(compute+charged) {
			return false
		}
		plan.Admission.Requests = append(plan.Admission.Requests, r.ID)
		plan.TokenCaps = append(plan.TokenCaps, DecisionTokenCap{Request: r.ID, Tokens: q})
		plan.Restores = append(plan.Restores, DecisionRestore{Request: r.ID, MaxPrefixBlocks: state.RecoverablePrefixBlocks})
		selected[r.ID] = true
		for _, key := range keys {
			plannedLoads[key] = true
		}
		load += l
		compute += charged
		budget -= charged
		slots--
		return true
	}
	absorb := func() {
		if !p.bundleHits {
			return
		}
		for _, r := range waiting {
			if selected[r.ID] {
				continue
			}
			keys, usable := decisionRestoreKeys(states[r.ID])
			covered := usable && len(keys) > 0
			for _, key := range keys {
				covered = covered && plannedLoads[key]
			}
			if covered {
				consider(r, true)
			}
		}
	}
	for _, r := range waiting {
		if selected[r.ID] {
			continue
		}
		if !consider(r, len(plan.Admission.Requests) == 0) {
			deferred = append(deferred, r)
		} else {
			absorb()
		}
	}
	// Fill unused room in original arrival order even if the ratio remains high.
	for _, r := range deferred {
		if consider(r, true) {
			absorb()
		}
	}
	plan.QueueOrder = append(plan.QueueOrder, plan.Admission.Requests...)
	for _, r := range waiting {
		if !selected[r.ID] {
			plan.QueueOrder = append(plan.QueueOrder, r.ID)
		}
	}
	return plan
}

func (*BalancedBatchPolicy) Observe(DecisionFeedback) {}
