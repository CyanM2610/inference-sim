package policylab

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type onceDeferralPolicy struct{ held bool }

func (p *onceDeferralPolicy) Decide(v sim.DecisionView) sim.DecisionPlan {
	plan := sim.DecisionPlan{Version: v.Version}
	var decode bool
	for _, r := range v.Running {
		decode = decode || r.EmittedTokens > 0
	}
	if !p.held && decode {
		for _, r := range v.Running {
			if r.PrefillDeferrable {
				plan.PrefillDeferrals = append(plan.PrefillDeferrals, r.ID)
			}
		}
		if len(plan.PrefillDeferrals) > 0 {
			plan.BatchTokenCap = 1
		}
	}
	return plan
}
func (p *onceDeferralPolicy) Observe(f sim.DecisionFeedback) {
	for _, r := range f.Unselected {
		if r.Reason == "policy_prefill_deferred" {
			p.held = true
		}
	}
}

func TestPrefillDeferralPublicFactoryControlsWorkAndRoundTrips(t *testing.T) {
	c := labConfig(1, "hbm")
	c.Instances[0].HBMBlocks = 32
	c.MaxBatchTokens, c.PrefillChunk = 32, 16
	c.Requests = c.Requests[:2]
	c.Requests[0].Input = c.Requests[0].Input[:65]
	c.Requests[1].Input = c.Requests[1].Input[:17]
	c.Requests[0].Output = []sim.TokenID{9}
	c.Requests[1].Output = []sim.TokenID{9, 10, 11, 12, 13}
	for i := range c.Requests {
		c.Requests[i].At = 0
	}
	c.DecisionPolicy = &DecisionPolicyConfig{PrefillDeferrals: true}
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Config
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if !decoded.DecisionPolicy.PrefillDeferrals {
		t.Fatal("configuration lost new capability")
	}
	policy := &onceDeferralPolicy{}
	r, err := RunWithDecisionPolicy(decoded, func(string) sim.DecisionPolicy { return policy })
	if err != nil {
		t.Fatal(err)
	}
	if !policy.held || len(r.FinishedUS) != 2 || r.PrefillDeferralCostCoverage == "" {
		t.Fatal("factory did not execute deferral", r.FinishedUS)
	}
	var held int
	var work int64
	for _, d := range r.PolicyDecisions {
		for _, g := range d.Feedback.Grants {
			work += g.Tokens
		}
		if len(d.Plan.PrefillDeferrals) > 0 {
			held++
			if len(d.Feedback.Grants) != 1 || d.Feedback.Grants[0].Request != c.Requests[1].ID {
				t.Fatal("held prefill stole decode budget", d)
			}
		}
	}
	if held != 1 || work != 86 || r.HBM["instance_0"]["active_or_pinned"] != 0 {
		t.Fatal("deferral leaked ownership or recomputed tokens", held, work, r.HBM)
	}
	c.DecisionPolicy.QueueServiceCost = &QueueServiceCostConfig{}
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "prefill deferrals") {
		t.Fatal("unmeasured action reused calibrated cost", err)
	}
}
