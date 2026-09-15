package sim

import (
	"fmt"
	"math"
	"slices"
)

// PrefillWaitProbe appends one resident-prefill wait per selected request to a
// base policy's decisions. It is a mechanism probe, not a scheduling baseline.
// The runtime retains ownership of KV, queue transitions and expiry timers.
type PrefillWaitProbe struct {
	base              DecisionPolicy
	durationUS        int64
	minComputedTokens int64
	requests          map[string]bool
	issued            map[string]bool
}

// NewPrefillWaitProbe selects all requests when requests is nil; a nonnil empty
// slice selects none. A zero duration preserves every base decision unchanged.
// Construction and Reset also reset the base when it implements Reset().
func NewPrefillWaitProbe(base DecisionPolicy, durationUS int64, requests []string) (*PrefillWaitProbe, error) {
	return NewPrefillWaitProbeWithOptions(base, durationUS, requests, PrefillWaitProbeOptions{})

}

// PrefillWaitProbeOptions chooses a later observed progress boundary without
// changing the runtime's eligibility, transfer state or completion clock.
type PrefillWaitProbeOptions struct {
	MinComputedTokens int64
}

func NewPrefillWaitProbeWithOptions(base DecisionPolicy, durationUS int64, requests []string, options PrefillWaitProbeOptions) (*PrefillWaitProbe, error) {
	if base == nil || durationUS < 0 || options.MinComputedTokens < 0 {
		return nil, fmt.Errorf("prefill wait probe requires a base policy and nonnegative duration/progress threshold")
	}
	p := &PrefillWaitProbe{base: base, durationUS: durationUS, minComputedTokens: options.MinComputedTokens}
	if requests != nil {
		p.requests = make(map[string]bool, len(requests))
		for _, id := range requests {
			p.requests[id] = true
		}
	}
	p.Reset()
	return p, nil
}

// Reset begins a fresh probe run. Issuance is remembered independently of
// feedback, expiry and requeue until this method is called.
func (p *PrefillWaitProbe) Reset() {
	p.issued = map[string]bool{}
	if base, ok := p.base.(interface{ Reset() }); ok {
		base.Reset()
	}
}

func (p *PrefillWaitProbe) Decide(v DecisionView) DecisionPlan {
	if !v.Capabilities.PrefillWaits {
		panic("prefill wait probe requires resident wait capability")
	}
	plan := p.base.Decide(v)
	if p.durationUS == 0 {
		return plan
	}
	preempted := make(map[string]bool, len(plan.Preemptions))
	for _, id := range plan.Preemptions {
		preempted[id] = true
	}
	// The only base-plan field changed here is Waits. Detach its backing array
	// before appending so a base policy can safely retain its own plan.
	plan.Waits = slices.Clone(plan.Waits)
	for _, r := range v.Running {
		if !r.PrefillWaitable || r.ComputedTokens < p.minComputedTokens || p.issued[r.ID] || preempted[r.ID] || p.requests != nil && !p.requests[r.ID] {
			continue
		}
		if v.NowUS > math.MaxInt64-p.durationUS {
			panic("prefill wait probe deadline overflows")
		}
		plan.Waits = append(plan.Waits, DecisionWait{Request: r.ID, UntilUS: v.NowUS + p.durationUS})
		p.issued[r.ID] = true
	}
	return plan
}

func (p *PrefillWaitProbe) Observe(f DecisionFeedback) { p.base.Observe(f) }
