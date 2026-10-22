package policylab

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

type DecisionPolicyConfig struct {
	QueueOrder      string                  `json:"queue_order,omitempty"`
	PrefillTokenCap int64                   `json:"prefill_token_cap,omitempty"`
	ExtraCost       *sim.LinearDecisionCost `json:"extra_cost,omitempty"`
}

// decisionAdapter retains the existing queue observation at the same boundary
// for baseline comparisons. Metrics/resource operations stay in the runtime.
type decisionAdapter struct {
	policy   sim.DecisionPolicy
	order    string
	instance string
	sink     kv.PeerSink
}

func (a *decisionAdapter) Decide(v sim.DecisionView) sim.DecisionPlan {
	p := a.policy.Decide(v)
	if a.sink != nil && len(v.Waiting) > 1 {
		ids := append([]string(nil), p.QueueOrder...)
		if p.QueueOrder == nil {
			for _, r := range v.Waiting {
				ids = append(ids, r.ID)
			}
		}
		a.sink(kv.PeerRecord{Time: v.NowUS, Name: "request_queue_order", Instance: a.instance, Reason: a.order, Requests: ids})
	}
	return p
}
func (a *decisionAdapter) Observe(f sim.DecisionFeedback) { a.policy.Observe(f) }
