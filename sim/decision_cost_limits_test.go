package sim

import "testing"

func TestDecisionCostEnvelopeWarnsOnRangeAndRejectsUnsupportedStates(t *testing.T) {
	v := producerView()
	v.KV.TransfersKnown = true
	v.Capabilities.BatchPrefixReuseKnown = true
	v.Capabilities.BatchPrefixReuse = true
	c := LinearDecisionCost{FixedUS: 5, Provenance: "bounded component fixture", Limits: &DecisionCostLimits{MaxVisibleRequests: 3, MaxPendingTransfers: 2, MaxInputTokens: 64, RequirePrefixProducers: true, RequireBatchPrefixReuse: true}}
	if r, err := c.Estimate(v, DecisionPlan{}); err != nil || r.ExtraUS != 5 {
		t.Fatal(r, err)
	}
	for _, change := range []func(*DecisionView){
		func(v *DecisionView) { v.KV.TransfersKnown = false },
		func(v *DecisionView) { v.Capabilities.BatchPrefixReuseKnown = false },
		func(v *DecisionView) { v.Capabilities.BatchPrefixReuse = false },
		func(v *DecisionView) { v.Capabilities.PrefixProducers = false },
	} {
		bad := cloneDecisionView(v)
		change(&bad)
		if _, err := c.Estimate(bad, DecisionPlan{}); err == nil {
			t.Fatal("cost profile silently extrapolated", bad)
		}
	}
	for _, change := range []func(*DecisionView){
		func(v *DecisionView) { v.Waiting = append(v.Waiting, DecisionRequest{ID: "extra", InputTokens: 64}) },
		func(v *DecisionView) { v.Waiting[0].InputTokens = 65 },
		func(v *DecisionView) { v.KV.PendingTransfers = make([]DecisionTransfer, 3) },
	} {
		warned := false
		c.SetProfileWarningObserver(func(_ string, observed, bound int64) { warned = observed > bound })
		wide := cloneDecisionView(v)
		change(&wide)
		if fee, err := c.Estimate(wide, DecisionPlan{}); err != nil || fee.ExtraUS != 5 || !warned {
			t.Fatal("extrapolation rejected or silent", fee, err)
		}
	}
	c.Limits.MaxVisibleRequests = 0
	if c.Validate() == nil {
		t.Fatal("invalid envelope accepted")
	}
	c.Limits = nil
	v.KV.TransfersKnown = false
	if _, err := c.Estimate(v, DecisionPlan{}); err != nil {
		t.Fatal("legacy unbounded fixed sensitivity changed", err)
	}
}
