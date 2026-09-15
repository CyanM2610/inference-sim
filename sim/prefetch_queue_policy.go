package sim

import (
	"fmt"
	"sort"
)

// PrefetchQueuePolicy is a one-attempt prefix-warming example. It retains queue
// order and ordinary admission, so blocked/no-op promotions can fall back to
// computation while submitted loads use the runtime's dependency checks.
// It is neither Cascade nor Strata; it exercises their required separation
// between initiating a load and granting a request's computation.
type PrefetchQueuePolicy struct {
	base      *QueueDecisionPolicy
	attempted map[string]bool
	prefetch  bool
}

func NewPrefetchQueuePolicy(order string) (*PrefetchQueuePolicy, error) {
	return NewPrefixQueuePolicy(order, true)
}

// NewPrefixQueuePolicy keeps queue semantics identical when comparing demand
// restoration with prefetching. Resource management stays in the runtime.
func NewPrefixQueuePolicy(order string, prefetch bool) (*PrefetchQueuePolicy, error) {
	base, err := NewQueueDecisionPolicy(order, 0)
	if err != nil {
		return nil, err
	}
	return &PrefetchQueuePolicy{base: base, attempted: map[string]bool{}, prefetch: prefetch}, nil
}

func (p *PrefetchQueuePolicy) Decide(v DecisionView) DecisionPlan {
	if p.prefetch && (!v.Capabilities.Promotions || !v.KV.PrefixStateKnown || v.BlockTokens <= 0) {
		panic("prefetch queue requires promotion and prefix-state capabilities")
	}
	ordered := v
	ordered.Waiting = append([]DecisionRequest(nil), v.Waiting...)
	if p.base.Order == "fcfs" {
		// Native waiting/skipped queues can arrive in a different physical
		// order. This example specifies FCFS by visible arrival time and ID.
		sort.SliceStable(ordered.Waiting, func(i, j int) bool {
			a, b := ordered.Waiting[i], ordered.Waiting[j]
			if a.ArrivalUS != b.ArrivalUS {
				return a.ArrivalUS < b.ArrivalUS
			}
			return a.ID < b.ID
		})
	}
	plan := p.base.Decide(ordered)
	if !p.prefetch {
		return plan
	}
	live := map[string]bool{}
	for _, group := range [][]DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			live[r.ID] = true
		}
	}
	for id := range p.attempted {
		if !live[id] {
			delete(p.attempted, id)
		}
	}
	states := map[string]DecisionKVRequest{}
	for _, state := range v.KV.Requests {
		states[state.ID] = state
	}
	byID := map[string]DecisionRequest{}
	for _, r := range v.Waiting {
		byID[r.ID] = r
	}
	for _, id := range plan.QueueOrder {
		r := byID[id]
		state, ok := states[r.ID]
		if !ok {
			panic(fmt.Sprintf("prefetch queue lacks prefix state for %q", r.ID))
		}
		endpoint := min(state.RecoverablePrefixBlocks, max(0, r.InputTokens-1)/v.BlockTokens)
		if !p.attempted[r.ID] && r.ComputedTokens == 0 && r.EmittedTokens == 0 && !state.TransferPending && !state.Deferred && endpoint > state.LocalPrefixBlocks {
			plan.Promotions = append(plan.Promotions, DecisionPromotion{Request: r.ID, MaxPrefixBlocks: endpoint})
		}
	}
	return plan
}

func (p *PrefetchQueuePolicy) Observe(f DecisionFeedback) {
	if f.Status == "applied" {
		for _, outcome := range f.Promotions {
			p.attempted[outcome.Request] = true
		}
	}
}
