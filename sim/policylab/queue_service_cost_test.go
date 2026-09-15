package policylab

import (
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func queueCostFixture(rate float64) Config {
	c := budgetLabConfig()
	c.DecisionPolicy.BudgetRestore.Mode = "budget_queues"
	c.DecisionPolicy.PrefillPreemption, c.DecisionPolicy.CapacityPreemption = true, true
	c.DecisionPolicy.CapacityReservationView, c.DecisionPolicy.ControlSteps, c.DecisionPolicy.ExecutionWork = true, true, true
	c.DecodeCapacityReservation = true
	c.BatchCost = &BatchCostConfig{CoefficientsUS: []float64{1, 0, 0, 0}, Provenance: "synthetic base"}
	for i := range c.Requests {
		c.Requests[i].MaxOutputTokens = 1
	}
	p := &QueueServiceCostConfig{Family: "budget_queues", EstimatorFamily: "plain_four", Shape: serviceShape(c),
		HBMBlocks: c.Instances[0].HBMBlocks, MaxInputTokens: 128, MaxOutputTokens: 1,
		Provenance: "synthetic phase boundaries", Coverage: "functional fixture only", Stages: map[string]QueueServiceCurve{}}
	for name, n := range queueStageSizes {
		x := QueueServiceCurve{CoefficientsUS: make([]float64, n), MaxCounts: make([]int64, n)}
		x.CoefficientsUS[0] = rate
		for i := range x.MaxCounts {
			x.MaxCounts[i] = 1000000
		}
		if name == "action" {
			x.CoefficientsUS, x.MaxCounts = nil, []int64{0}
		}
		p.Stages[name] = x
	}
	c.DecisionPolicy.QueueServiceCost = p
	return c
}

func TestQueueServiceZeroPreservesEventsAndPositiveFeesReachRuntime(t *testing.T) {
	c := queueCostFixture(0)
	p := c.DecisionPolicy.QueueServiceCost
	c.DecisionPolicy.QueueServiceCost = nil
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.QueueServiceCost = p
	zero, err := Run(c)
	if err != nil || !reflect.DeepEqual(base.Events, zero.Events) || !reflect.DeepEqual(base.FirstTokenUS, zero.FirstTokenUS) || !reflect.DeepEqual(base.FinishedUS, zero.FinishedUS) {
		t.Fatal("zero queue fees changed physical execution", err)
	}
	for name, x := range p.Stages {
		if x.CoefficientsUS != nil {
			x.CoefficientsUS[0] = 1.2
			p.Stages[name] = x
		}
	}
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(paid.FinishedUS) != len(c.Requests) || paid.FirstTokenUS["warm"] <= zero.FirstTokenUS["warm"] || len(paid.HostServices) == 0 {
		t.Fatal("queue fee did not affect real service or lost requests")
	}
	controls := 0
	for _, r := range paid.PolicyDecisions {
		if r.Service == nil || r.ExecutionService == nil || r.Service.ExtraUS != 4 || r.ExecutionService.ExtraUS != 3 || r.Feedback.AtUS != r.ExecutionService.EndUS || r.ExecutionService.StartUS != r.Service.EndUS {
			t.Fatal("wrong aggregation/rounding or feedback delivered before service", r)
		}
		if r.ControlStep {
			controls++
		}
	}
	if controls == 0 || paid.HBM["instance_0"]["active_or_pinned"] != 0 {
		t.Fatal("missing idle control coverage or leaked memory")
	}
}

func TestQueueServiceRejectsUnmeasuredActionsAndDuplicateFees(t *testing.T) {
	c := queueCostFixture(1)
	m, err := newQueueServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	_, err = m.EstimateExecutionOutcome(sim.DecisionView{}, sim.DecisionPlan{}, sim.ExecutionOutcome{
		Work: sim.BatchExecutionWork{TokenChecksKnown: true}, PreemptionCount: 1, Preemptions: []sim.DecisionPreemptionOutcome{{Request: "victim", Status: "requeued"}}})
	if err == nil {
		t.Fatal("unmeasured action silently became free")
	}
	c.DecisionPolicy.ExtraCost = &sim.LinearDecisionCost{FixedUS: 2, Provenance: "duplicate"}
	if c.Validate() == nil {
		t.Fatal("controller charged twice")
	}
	c.DecisionPolicy.ExtraCost = nil
	c.BatchCost.EnqueueUS = 1
	if c.Validate() == nil {
		t.Fatal("full registration charged on top of enqueue")
	}
	c.BatchCost.EnqueueUS = 0
	c.DecisionPolicy.QueueServiceCost.EstimatorFamily = "worker_six"
	if c.Validate() == nil {
		t.Fatal("different estimator implementation reused the fee profile")
	}
}

func TestQueueServiceDoesNotSilentlyPriceBudgetReestimation(t *testing.T) {
	c := queueCostFixture(1)
	c.DecisionPolicy.BudgetRestore.ReestimateOnDecodeDrop = true
	if c.Validate() == nil {
		t.Fatal("unmeasured budget refresh reused queue service profile")
	}
	c.DecisionPolicy.QueueServiceCost = nil
	if err := c.Validate(); err != nil {
		t.Fatal("functional refresh configuration rejected", err)
	}
	c.DecisionPolicy.BudgetRestore.Mode = "budget_pressure"
	if c.Validate() == nil {
		t.Fatal("non-queue mode ignored refresh option")
	}
}

func TestQueueServiceRetainsBaseStagesAndAddsExplicitRefreshService(t *testing.T) {
	c := queueCostFixture(1)
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.BudgetRestore.ReestimateOnDecodeDrop = true
	c.DecisionPolicy.QueueServiceCost.ReestimateExtraUS = 20
	c.DecisionPolicy.QueueServiceCost.ReestimateProvenance = "declared sensitivity increment; base stages exclude refresh work"
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(paid.FinishedUS) != len(c.Requests) || paid.FirstTokenUS["warm"] <= base.FirstTokenUS["warm"] {
		t.Fatal("additional service did not reach actual runtime")
	}
	for _, d := range paid.PolicyDecisions {
		if d.Service == nil || d.Service.ExtraUS != 23 || d.ExecutionService == nil || d.ExecutionService.ExtraUS != 2 ||
			d.ExecutionService.StartUS != d.Service.EndUS || d.Feedback.AtUS != d.ExecutionService.EndUS {
			t.Fatal("refresh replaced/doubled a base stage or escaped service ordering", d)
		}
	}
	registrations := 0
	for _, h := range paid.HostServices {
		// Added CPU service can create additional idle completions. Verify
		// the unchanged per-call model, not an unchanged number of events.
		if h.Service.ExtraUS != 1 {
			t.Fatal("refresh altered the registration/completion unit fee")
		}
		if h.Work.Stage == "registration" {
			registrations++
		}
	}
	if registrations != len(c.Requests) {
		t.Fatal("request registration lost or duplicated")
	}
	c.DecisionPolicy.QueueServiceCost.ReestimateExtraUS = math.MaxInt64
	m, err := newQueueServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	first := paid.PolicyDecisions[0]
	if _, err := m.Estimate(first.View, first.Plan); err == nil {
		t.Fatal("additional service overflow accepted")
	}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.DecisionPolicy.QueueServiceCost.ReestimateExtraUS = 0 },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost.ReestimateProvenance = " " },
		func(c *Config) { c.DecisionPolicy.BudgetRestore.ReestimateOnDecodeDrop = false },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost.HBMBlocks = 0 },
	} {
		bad := queueCostFixture(1)
		bad.DecisionPolicy.BudgetRestore.ReestimateOnDecodeDrop = true
		bad.DecisionPolicy.QueueServiceCost.ReestimateExtraUS, bad.DecisionPolicy.QueueServiceCost.ReestimateProvenance = 20, "declared"
		mutate(&bad)
		if bad.Validate() == nil {
			t.Fatal("unsupported refresh service or base geometry accepted")
		}
	}
}
