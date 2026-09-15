package policylab

import (
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func waitDecisionFixture(c Config) *WaitDecisionCostConfig {
	family := "basic"
	if c.DecisionPolicy.CapacityPreemption {
		family = "capacity"
	}
	return &WaitDecisionCostConfig{Family: family, RatesUS: []int64{157, 14, 0, 31, 30}, MaxCounts: []int64{1, 32, 64, 32, 32},
		MaxInputTokens: 64, MaxOutputTokens: 4, HBMBlocks: c.Instances[0].HBMBlocks, Shape: serviceShape(c), Provenance: "synthetic wait decision boundary test"}
}

func TestWaitDecisionPricesAdmissionAndResidentUpdatesIncludingClear(t *testing.T) {
	c := pressureBudgetConfig()
	c.DecisionPolicy.PrefillWaits = true
	c.DecisionPolicy.WaitDecisionCost = waitDecisionFixture(c)
	m, err := newWaitDecisionCost(c)
	if err != nil {
		t.Fatal(err)
	}
	v := sim.DecisionView{HBMCapacityBlocks: c.Instances[0].HBMBlocks,
		Capabilities: sim.DecisionCapabilities{PrefillWaits: true, CapacityPreemption: true},
		Waiting:      []sim.DecisionRequest{{ID: "admission", InputTokens: 32, ClientOutputLimit: 1}, {ID: "other", InputTokens: 32, ClientOutputLimit: 1}},
		Running:      []sim.DecisionRequest{{ID: "resident", InputTokens: 32, ComputedTokens: 16, ClientOutputLimit: 1}},
		KV:           sim.DecisionKVState{TransfersKnown: true, PendingTransfers: []sim.DecisionTransfer{{}}}}
	plan := sim.DecisionPlan{Waits: []sim.DecisionWait{{Request: "admission", UntilUS: 0}, {Request: "resident", UntilUS: 1000}}}
	// Both native wait updates cost service, even the admission deadline clear.
	fee, err := m.Estimate(v, plan)
	if err != nil || fee.ExtraUS != 291 {
		t.Fatal("waiting update was lost or clear was free", fee, err)
	}
	c.DecisionPolicy.WaitDecisionCost.RatesUS[0] = 999
	c.DecisionPolicy.WaitDecisionCost.MaxCounts[3] = 0
	if again, err := m.Estimate(v, plan); err != nil || again != fee {
		t.Fatal("caller changed installed coefficients or bounds", again, err)
	}
	v.Estimates = &sim.DecisionEstimates{Requests: []sim.DecisionRequestEstimate{{Choices: []sim.DecisionRestoreEstimate{{}, {}}}}}
	m.config.MaxCounts[2] = 1
	if _, err := m.Estimate(v, plan); err == nil {
		t.Fatal("zero-rate restore feature escaped coverage limits")
	}
	v.Estimates = nil
	v.KV.TransfersKnown = false
	if _, err := m.Estimate(v, plan); err == nil {
		t.Fatal("unknown pending transfers priced as zero")
	}
	v.KV.TransfersKnown = true
	m.config.RatesUS[0] = math.MaxInt64
	if _, err := m.Estimate(v, plan); err == nil {
		t.Fatal("decision fee overflow accepted")
	}
}

func TestWaitDecisionRejectsOverlappingAndUnmeasuredProfiles(t *testing.T) {
	c := pressureBudgetConfig()
	c.DecisionPolicy.PrefillWaits = true
	c.DecisionPolicy.WaitDecisionCost = waitDecisionFixture(c)
	c.DecisionPolicy.ExtraCost = &sim.LinearDecisionCost{Provenance: "old"}
	if err := c.Validate(); err == nil {
		t.Fatal("two decision cost models accepted")
	}
	c.DecisionPolicy.ExtraCost = nil
	c.DecisionPolicy.PrefixSharing = true
	if _, err := newWaitDecisionCost(c); err == nil {
		t.Fatal("unmeasured sharing wrapper accepted")
	}
	c.DecisionPolicy.PrefixSharing = false
	c.DecisionPolicy.WaitDecisionCost.Family = "basic"
	if _, err := newWaitDecisionCost(c); err == nil {
		t.Fatal("wrong controller family accepted")
	}
}

func TestWaitDecisionZeroPreservesEventsAndServicePrecedesExecution(t *testing.T) {
	c := restoreLabConfig(nil)
	c.PrefillChunk = 16
	c.DecisionPolicy.PrefillWaits = true
	c.DecisionPolicy.ControlSteps = true
	c.DecisionPolicy.PrefillWaitProbe = &PrefillWaitProbeConfig{DurationUS: 1000, Requests: []string{"warm"}}
	for i := range c.Requests {
		c.Requests[i].MaxOutputTokens = 1
	}
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.WaitDecisionCost = waitDecisionFixture(c)
	c.DecisionPolicy.WaitDecisionCost.RatesUS = []int64{0, 0, 0, 0, 0}
	zero, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Events, zero.Events) || !reflect.DeepEqual(base.FirstTokenUS, zero.FirstTokenUS) || !reflect.DeepEqual(base.HBM, zero.HBM) {
		t.Fatal("zero wait decision service changed execution")
	}
	c.DecisionPolicy.WaitDecisionCost.RatesUS = []int64{5, 2, 3, 7, 4}
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(paid.FinishedUS) != len(c.Requests) || paid.FirstTokenUS["warm"] <= zero.FirstTokenUS["warm"] {
		t.Fatal("paid decisions did not delay or complete actual execution")
	}
	waited := false
	for _, r := range paid.PolicyDecisions {
		choices := 0
		if r.View.Estimates != nil {
			for _, estimate := range r.View.Estimates.Requests {
				choices += len(estimate.Choices)
			}
		}
		want := int64(5 + 2*(len(r.View.Waiting)+len(r.View.Running)) + 3*choices + 7*len(r.Plan.Waits) + 4*len(r.View.KV.PendingTransfers))
		if r.Service == nil || r.Service.ExtraUS != want || r.Service.EndUS-r.Service.StartUS != want || r.Feedback.AtUS < r.Service.EndUS {
			t.Fatal("decision was applied without paying its service", r)
		}
		waited = waited || len(r.Plan.Waits) > 0
	}
	if !waited {
		t.Fatal("no wait update exercised")
	}
}
