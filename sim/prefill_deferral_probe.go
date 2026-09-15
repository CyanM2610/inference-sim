package sim

import "fmt"

// PrefillDeferralProbe exercises a bounded number of actual round holds while
// a decode can execute. It is a mechanism probe, not a Strata baseline.
type PrefillDeferralProbe struct {
	base                   DecisionPolicy
	maxRounds, minComputed int64
	requests               map[string]bool
	counts                 map[string]int64
	planned                map[string]bool
	version                uint64
}

func NewPrefillDeferralProbe(base DecisionPolicy, rounds, minComputed int64, requests []string) (DecisionPolicy, error) {
	if base == nil || rounds < 0 || minComputed < 0 {
		return nil, fmt.Errorf("deferral probe requires a base and nonnegative round/progress limits")
	}
	p := &PrefillDeferralProbe{base: base, maxRounds: rounds, minComputed: minComputed}
	if requests != nil {
		p.requests = map[string]bool{}
		for _, id := range requests {
			if id == "" || p.requests[id] {
				return nil, fmt.Errorf("deferral probe requires distinct nonempty request IDs")
			}
			p.requests[id] = true
		}
	}
	p.Reset()
	if listener, ok := base.(DecisionEventListener); ok {
		return &prefillDeferralProbeEvents{PrefillDeferralProbe: p, listener: listener}, nil
	}
	return p, nil
}

func (p *PrefillDeferralProbe) Reset() {
	p.counts = map[string]int64{}
	p.planned = nil
	p.version = 0
	if base, ok := p.base.(interface{ Reset() }); ok {
		base.Reset()
	}
}

func (p *PrefillDeferralProbe) Decide(v DecisionView) DecisionPlan {
	if !v.Capabilities.PrefillDeferrals {
		panic("deferral probe requires per-round prefill capability")
	}
	plan := p.base.Decide(v)
	p.version = v.Version
	p.planned = map[string]bool{}
	if p.maxRounds == 0 {
		return plan
	}
	blocked := map[string]bool{}
	for _, r := range v.KV.Requests {
		blocked[r.ID] = r.TransferPending || r.Deferred
	}
	decode := false
	for _, r := range v.Running {
		decode = decode || r.State == StateRunning && r.EmittedTokens > 0 && r.ComputedTokens >= max(r.InputTokens, r.RecomputeUntilTokens) && !blocked[r.ID]
	}
	if !decode {
		return plan
	}
	excluded := map[string]bool{}
	for _, id := range plan.Preemptions {
		excluded[id] = true
	}
	for _, id := range plan.PrefillDeferrals {
		excluded[id] = true
	}
	plan.PrefillDeferrals = append([]string(nil), plan.PrefillDeferrals...)
	for _, r := range v.Running {
		if r.PrefillDeferrable && r.ComputedTokens >= p.minComputed && p.counts[r.ID] < p.maxRounds && !excluded[r.ID] && (p.requests == nil || p.requests[r.ID]) {
			plan.PrefillDeferrals = append(plan.PrefillDeferrals, r.ID)
			p.planned[r.ID] = true
		}
	}
	return plan
}

func (p *PrefillDeferralProbe) Observe(f DecisionFeedback) {
	if f.Version != p.version {
		panic("deferral probe feedback has the wrong decision version")
	}
	p.base.Observe(f)
	if f.Status == "applied" {
		for _, r := range f.Unselected {
			if p.planned[r.Request] && r.Reason == "policy_prefill_deferred" {
				p.counts[r.Request]++
				delete(p.planned, r.Request)
			}
		}
	}
	p.planned = nil
}

func (*PrefillDeferralProbe) AdditionalCostCoverage() string {
	return "prefill_deferral_probe_not_independently_calibrated"
}

type prefillDeferralProbeEvents struct {
	*PrefillDeferralProbe
	listener DecisionEventListener
}

func (p *prefillDeferralProbeEvents) OnEvent(e DecisionEvent) { p.listener.OnEvent(e) }
