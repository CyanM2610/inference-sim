package sim

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func bundleView() DecisionView {
	v := balancedView()
	for i := range v.Waiting {
		v.Waiting[i].InputTokens = 257
	}
	v.KV.PrefixSharingKnown, v.KV.TransfersKnown = true, true
	for i := range v.KV.Requests {
		for j := 0; j < 16; j++ {
			key := fmt.Sprintf("shared-%d", j)
			if i == 2 {
				key = fmt.Sprintf("cold-%d", j)
			}
			v.KV.Requests[i].PrefixBlocks = append(v.KV.Requests[i].PrefixBlocks, DecisionPrefixBlock{ID: key})
		}
	}
	return v
}

func TestBundleAbsorptionRespectsBudgetAndOpaqueIdentity(t *testing.T) {
	v := bundleView()
	p, _ := NewBundleBalancedPolicy(100, 128, 2, 0)
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.Admission.Requests, []string{"a-hot", "b-hot"}) || !reflect.DeepEqual(plan.TokenCaps, []DecisionTokenCap{{"a-hot", 1}, {"b-hot", 1}}) {
		t.Fatal("bundle did not replace compute partner", plan)
	}
	before := cloneDecisionView(v)
	for i := range v.KV.Requests {
		for j := range v.KV.Requests[i].PrefixBlocks {
			v.KV.Requests[i].PrefixBlocks[j].ID = "renamed-" + v.KV.Requests[i].PrefixBlocks[j].ID
		}
	}
	if !reflect.DeepEqual(plan, p.Decide(v)) || before.KV.Requests[0].PrefixBlocks[0].ID != "shared-0" {
		t.Fatal("opaque identity affected priority or nested snapshot aliases")
	}
	p.TokenBudget = 1
	if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"a-hot"}) {
		t.Fatal("free-load bundle escaped compute budget", got)
	}
}

func TestBundlePartialSharingAndPendingDependencies(t *testing.T) {
	v := bundleView()
	for i := 8; i < 16; i++ {
		v.KV.Requests[1].PrefixBlocks[i].ID = fmt.Sprintf("other-%d", i)
	}
	p, _ := NewBundleBalancedPolicy(200, 128, 2, 0)
	if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"a-hot", "b-hot"}) {
		t.Fatal("partial overlap was charged twice", got)
	}
	// A different request owns this pending prefix; the follower itself has no
	// transfer_pending flag. It must still wait for scheduler adoption.
	v.KV.Requests[0].PrefixBlocks[0].LoadPending = true
	v.KV.Requests[1].PrefixBlocks[0].LoadPending = true
	if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"c-cold"}) {
		t.Fatal("pending shared load became a usable hit", got)
	}
	// Known identity without ready remote content is not a bundle hit.
	v.KV.Requests[0].RecoverablePrefixBlocks = 0
	v.KV.Requests[1].RecoverablePrefixBlocks = 0
	if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"a-hot"}) {
		t.Fatal("future shared input treated as ready KV", got)
	}
}

func TestPrefixSharingCostRejectsOldCoverageAndCountsObservationWork(t *testing.T) {
	v := bundleView()
	c := LinearDecisionCost{Provenance: "synthetic sharing fee", PerPrefixBlockUS: 2,
		Limits: &DecisionCostLimits{MaxVisibleRequests: 3, MaxInputTokens: 257}}
	if _, err := c.Estimate(v, DecisionPlan{}); err == nil {
		t.Fatal("old profile silently covers new sharing work")
	}
	c.Limits.RequirePrefixSharing, c.Limits.MaxPrefixBlocks = true, 48
	e, err := c.Estimate(v, DecisionPlan{})
	if err != nil || e.ExtraUS != 96 {
		t.Fatal("sharing work should charge observed entries, not unique content", e, err)
	}
	c.Limits.MaxPrefixBlocks = 47
	warned := false
	c.SetProfileWarningObserver(func(parameter string, observed, maximum int64) {
		warned = parameter == "prefix_blocks" && observed == 48 && maximum == 47
	})
	if cost, err := c.Estimate(v, DecisionPlan{}); err != nil || cost.ExtraUS != 96 || !warned {
		t.Fatal("sharing extrapolation clipped, rejected or silent", cost, err)
	}
	c.Limits = nil
	c.PerPrefixBlockUS = math.MaxInt64
	if _, err := c.Estimate(v, DecisionPlan{}); err == nil {
		t.Fatal("sharing work cost overflow accepted")
	}
}

func TestBundleRestoreExcludesFinalLogitsBlock(t *testing.T) {
	v := bundleView()
	for i := range v.Waiting {
		v.Waiting[i].InputTokens = 256
	}
	p, _ := NewBundleBalancedPolicy(100, 128, 2, 0)
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.Restores, []DecisionRestore{{"a-hot", 15}, {"b-hot", 15}}) || !reflect.DeepEqual(plan.TokenCaps, []DecisionTokenCap{{"a-hot", 16}, {"b-hot", 16}}) {
		t.Fatal("final logits block was assumed shareable", plan)
	}
}
