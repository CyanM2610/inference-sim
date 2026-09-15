package sim

import (
	"math"
	"testing"
)

func TestPromotionDecisionCostChargesAttemptsAndBoundsRetention(t *testing.T) {
	v := DecisionView{BlockTokens: 16, Waiting: []DecisionRequest{{ID: "a", InputTokens: 65}, {ID: "b", InputTokens: 33}},
		Capabilities: DecisionCapabilities{Promotions: true, PromotionRetention: "request"}, KV: DecisionKVState{TransfersKnown: true}}
	p := DecisionPlan{Promotions: []DecisionPromotion{{Request: "a", MaxPrefixBlocks: 4}, {Request: "b", MaxPrefixBlocks: 2}}}
	c := LinearDecisionCost{FixedUS: 7, PerVisibleRequestUS: 3, PerPromotionUS: 11, PerPromotionBlockUS: 2, Provenance: "attempt cost fixture",
		Limits: &DecisionCostLimits{MaxVisibleRequests: 2, MaxPendingTransfers: 0, MaxInputTokens: 65, RequirePromotionRetention: "request"}}
	r, err := c.Estimate(v, p)
	if err != nil || r.ExtraUS != 47 {
		t.Fatal(r, err)
	}
	r, err = c.Estimate(v, DecisionPlan{})
	if err != nil || r.ExtraUS != 13 {
		t.Fatal("demand baseline acquired promotion fees", r, err)
	}
	v.Capabilities.PromotionRetention = "cache"
	if _, err = c.Estimate(v, p); err == nil {
		t.Fatal("retention mismatch silently profiled")
	}
	v.Capabilities.PromotionRetention = "request"
	c.Limits.MaxPromotionBlocks = 5
	warned := ""
	c.SetProfileWarningObserver(func(parameter string, _, _ int64) { warned = parameter })
	if fee, err := c.Estimate(v, p); err != nil || fee.ExtraUS != 47 || warned != "promotion_blocks" {
		t.Fatal("promotion block extrapolation rejected, clipped or silent", fee, err)
	}
	c.Limits.MaxPromotionBlocks = 6
	c.Limits.MaxPromotionActions = 1
	warned = ""
	if fee, err := c.Estimate(v, p); err != nil || fee.ExtraUS != 47 || warned != "promotion_actions" {
		t.Fatal("promotion count extrapolation rejected, clipped or silent", fee, err)
	}
	c.Limits.MaxPromotionActions = 2
	c.PerPromotionBlockUS = math.MaxInt64
	if _, err = c.Estimate(v, p); err == nil {
		t.Fatal("promotion cost overflow")
	}
	c.PerPromotionBlockUS = -1
	if c.Validate() == nil {
		t.Fatal("negative promotion cost")
	}
}

func TestDemandPrefixQueueKeepsExplicitFCFSWithoutKVActions(t *testing.T) {
	p, err := NewPrefixQueuePolicy("fcfs", false)
	if err != nil {
		t.Fatal(err)
	}
	v := DecisionView{Version: 3, Waiting: []DecisionRequest{{ID: "later", ArrivalUS: 20}, {ID: "first", ArrivalUS: 10}}}
	plan := p.Decide(v)
	if len(plan.QueueOrder) != 2 || plan.QueueOrder[0] != "first" || len(plan.Promotions) != 0 || len(plan.Restores) != 0 {
		t.Fatal(plan)
	}
	if v.Waiting[0].ID != "later" {
		t.Fatal("policy mutated detached caller snapshot")
	}
}
