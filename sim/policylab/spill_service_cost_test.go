package policylab

import (
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func spillCostFixture(mode string) Config {
	c := requestSpillConfig(mode)
	c.DecisionPolicy.BudgetRestore = &BudgetRestoreConfig{Mode: "budget_pressure", Guardband: 1}
	c.DecisionPolicy.CapacityPreemption, c.DecisionPolicy.CapacityReservationView = true, true
	c.DecodeCapacityReservation = true
	c.DecisionEstimates = &DecisionEstimateConfig{MaxDecodeContextTokens: 2048, Spill: true}
	c.MaxSequences, c.MaxBatchTokens = 2, 48
	c.Requests[0].TTFTSLOUS, c.Requests[1].TTFTSLOUS = 1000000, 1
	base := queueCostFixture(0)
	c.BatchCost = base.BatchCost
	p := base.DecisionPolicy.QueueServiceCost
	p.Family, p.Shape, p.HBMBlocks = "budget_pressure", serviceShape(c), c.Instances[0].HBMBlocks
	p.Stages["action"] = QueueServiceCurve{CoefficientsUS: []float64{0}, MaxCounts: []int64{1}}
	zero := int64(0)
	x := &SpillServiceCostConfig{Mode: mode, MaxRequests: 2, MaxSourceBlocks: 2, NativeBoundaryExtraUS: &zero,
		NativeBoundaryProvenance: "explicit synthetic lower endpoint", Provenance: "synthetic exclusive fees", Coverage: "functional test only",
		Stages: map[string]QueueServiceCurve{}}
	for name, size := range spillStageSizes(mode) {
		curve := QueueServiceCurve{CoefficientsUS: make([]float64, size), MaxCounts: make([]int64, size)}
		for i := range curve.MaxCounts {
			curve.MaxCounts[i] = 10000
		}
		x.Stages[name] = curve
	}
	p.Spill = x
	c.DecisionPolicy.QueueServiceCost = p
	return c
}

func TestSpillServiceUsesActualOutcomesAndKeepsDefaultExecution(t *testing.T) {
	for _, mode := range []string{"spill", "recompute"} {
		t.Run(mode, func(t *testing.T) {
			c := spillCostFixture(mode)
			profile := c.DecisionPolicy.QueueServiceCost
			c.DecisionPolicy.QueueServiceCost = nil
			before, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			c.DecisionPolicy.QueueServiceCost = profile
			zero, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before.Events, zero.Events) || !reflect.DeepEqual(before.FirstTokenUS, zero.FirstTokenUS) {
				t.Fatal("zero exclusive service changed physical baseline")
			}
			profile.Stages["action"] = QueueServiceCurve{CoefficientsUS: []float64{13}, MaxCounts: []int64{1}}
			profile.Spill.Stages["spill_prepare"] = QueueServiceCurve{CoefficientsUS: []float64{31, 11}, MaxCounts: []int64{1, 1}}
			profile.Spill.Stages["spill_store"] = QueueServiceCurve{CoefficientsUS: []float64{73}, MaxCounts: []int64{1}}
			*profile.Spill.NativeBoundaryExtraUS = 17
			paid, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			if len(paid.FinishedUS) != 2 || paid.FirstTokenUS["replacement"] <= zero.FirstTokenUS["replacement"] || paid.RequestSpillCostCoverage != profile.Spill.Coverage {
				t.Fatal("service failed to reach real completion or coverage")
			}
			actual, conditional := 0, 0
			for _, d := range paid.PolicyDecisions {
				if len(d.Feedback.PreemptionStorage) > 0 {
					actual++
					want := int64(13 + 11 + 17)
					if mode == "spill" {
						want = 13 + 31 + 73 + 17
					}
					if d.ExecutionService == nil || d.ExecutionService.ExtraUS != want || d.Feedback.AtUS != d.ExecutionService.EndUS {
						t.Fatal("actual action mispriced or feedback delivered before service", d.ExecutionService)
					}
				} else if len(d.Plan.PreemptionStorage) > 0 {
					conditional++
					if d.ExecutionService == nil || d.ExecutionService.ExtraUS != 0 {
						t.Fatal("unselected victim charged as a real action")
					}
				}
			}
			if actual != 1 || conditional == 0 {
				t.Fatal("fixture did not exercise both actual and conditional choices", actual, conditional)
			}
			for _, pool := range paid.Pools {
				if pool["reserved"] != 0 || pool["read_pins"] != 0 {
					t.Fatal("spill service leaked leases")
				}
			}
		})
	}
}

func TestSpillServiceRejectsUnmeasuredExecutionAndConfiguration(t *testing.T) {
	c := spillCostFixture("spill")
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	var active sim.DecisionRecord
	for _, d := range r.PolicyDecisions {
		if len(d.Feedback.PreemptionStorage) > 0 {
			active = d
		}
	}
	if len(active.Feedback.PreemptionStorage) != 1 {
		t.Fatal("missing actual victim")
	}
	warnings := &profileWarnings{}
	if _, err := spillExecutionCounts(active.View, active.Plan, active.Feedback, 1, profileReporter{warnings: warnings, component: "spill_service"}); err != nil {
		t.Fatal("valid larger spill rejected", err)
	}
	requireProfileWarning(t, warnings.snapshot(), "spill_service", "source_blocks")
	for _, mutate := range []func(*sim.DecisionFeedback){
		func(f *sim.DecisionFeedback) { f.PreemptionStorage = nil },
		func(f *sim.DecisionFeedback) { f.PreemptionStorage[0].Request = "other" },
		func(f *sim.DecisionFeedback) { f.PreemptionStorage[0].Status = "recompute_capacity" },
		func(f *sim.DecisionFeedback) { f.PreemptionStorage[0].NewBlocks-- },
		func(f *sim.DecisionFeedback) { f.PreemptionStorage[0].PendingBlocks++ },
	} {
		f := active.Feedback
		f.PreemptionStorage = append([]sim.RequestSpillOutcome(nil), f.PreemptionStorage...)
		mutate(&f)
		if _, err := SpillExecutionCounts(active.View, active.Plan, f, 2); err == nil {
			t.Fatal("unmeasured or missing storage outcome priced")
		}
	}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.DecisionPolicy.QueueServiceCost.Spill = nil },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost.Spill.NativeBoundaryExtraUS = nil },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost.Spill.NativeBoundaryProvenance = "" },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost.Spill.MaxRequests = 0 },
		func(c *Config) { c.DecisionPolicy.QueueServiceCost.Spill.Mode = "budget" },
		func(c *Config) { c.DecisionPolicy.RequestSpill = false },
		func(c *Config) { c.Mechanisms.BackgroundStoreMode = "continuous" },
		func(c *Config) { c.DecisionEstimates.Spill = false },
		func(c *Config) { delete(c.DecisionPolicy.QueueServiceCost.Spill.Stages, "spill_store") },
		func(c *Config) {
			c.DecisionPolicy.QueueServiceCost.Spill.Stages["spill_store"].CoefficientsUS[0] = math.NaN()
		},
	} {
		bad := spillCostFixture("spill")
		mutate(&bad)
		if bad.Validate() == nil {
			t.Fatal("unsupported spill fee configuration accepted")
		}
	}
	m, err := newQueueServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	*c.DecisionPolicy.QueueServiceCost.Spill.NativeBoundaryExtraUS = 999
	if *m.profile.Spill.NativeBoundaryExtraUS != 0 {
		t.Fatal("mutable residual allowance leaked into installed model")
	}
}

func TestSpillDecisionChargesExclusiveWorkAndRejectsCollinearScopeEscape(t *testing.T) {
	c := spillCostFixture("spill")
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	var d sim.DecisionRecord
	for _, row := range r.PolicyDecisions {
		if len(row.Plan.PreemptionStorage) > 0 {
			d = row
			break
		}
	}
	counts, err := SpillDecisionCounts(d.View, d.Plan, "budget")
	if err != nil {
		t.Fatal(err)
	}
	if counts["spill_estimate"][1] != 1 || counts["spill_policy_budget"][1] != 1 {
		t.Fatal("fixture lacks a budget target/choice")
	}
	plan := d.Plan
	plan.PreemptionStorage = nil
	if _, err := SpillDecisionCounts(d.View, plan, "budget"); err == nil {
		t.Fatal("collinear target/choice basis escaped measured relation")
	}
	x := c.DecisionPolicy.QueueServiceCost.Spill
	for _, name := range []string{"spill_view", "spill_estimate", "spill_plan", "spill_policy_spill"} {
		x.Stages[name].CoefficientsUS[0] = 1.2
	}
	m, err := newQueueServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	fee, err := m.Estimate(d.View, d.Plan)
	if err != nil || fee.ExtraUS != 5 {
		t.Fatal("exclusive pre-dispatch work missing or duplicated", fee, err)
	}
	*x.NativeBoundaryExtraUS = math.MaxInt64
	m, err = newQueueServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.addSpillService(sim.DecisionCostEstimate{}, map[string][]int64{"spill_prepare": {1, 0}, "spill_store": {1}}, true); err == nil {
		t.Fatal("spill fee overflow accepted")
	}
}
