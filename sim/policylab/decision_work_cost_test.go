package policylab

import (
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func serviceProfile(c Config) *RestoreServiceCostConfig {
	p := &RestoreServiceCostConfig{Shape: serviceShape(c), RatesUS: map[string]float64{}, MaxWork: map[string]int64{}, Provenance: "synthetic test", Coverage: "test components only"}
	p.Shape.MaxInputTokens = 4096
	for _, name := range restoreWorkNames {
		p.RatesUS[name] = 0
		p.MaxWork[name] = 100000
	}
	return p
}

func TestRestoreServiceChargesCandidateWorkAndWarnsOnExtrapolation(t *testing.T) {
	c := budgetLabConfig()
	c.profileWarnings = &profileWarnings{}
	c.DecisionPolicy.RestoreServiceCost = serviceProfile(c)
	c.DecisionPolicy.RestoreServiceCost.RatesUS["compute_steps"] = 2
	m, err := newRestoreServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	predictor, _ := newRestoreEstimator(c)
	v := sim.DecisionView{BlockTokens: c.BlockTokens, MaxBatchTokens: c.MaxBatchTokens, MaxSequences: c.MaxSequences, PrefillChunk: c.PrefillChunk,
		Waiting: []sim.DecisionRequest{{ID: "r", InputTokens: 32}}, KV: sim.DecisionKVState{TransfersKnown: true, PrefixStateKnown: true, Requests: []sim.DecisionKVRequest{{ID: "r"}}}}
	e, err := predictor.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	v.Estimates = &e
	a, err := m.Estimate(v, sim.DecisionPlan{})
	if err != nil {
		t.Fatal(err)
	}
	v.KV.Requests[0].RecoverablePrefixBlocks = 2
	e, err = predictor.Estimate(v)
	if err != nil {
		t.Fatal(err)
	}
	v.Estimates = &e
	b, err := m.Estimate(v, sim.DecisionPlan{})
	if err != nil || b.ExtraUS <= a.ExtraUS {
		t.Fatalf("candidate work not charged: %v %v %v", a, b, err)
	}
	w, _ := RestoreDecisionWork(c, v)
	if b.ExtraUS != 2*w["compute_steps"] {
		t.Fatal("service not based on predictor work")
	}
	m.config.MaxWork["candidates"] = 0
	if extrapolated, err := m.Estimate(v, sim.DecisionPlan{}); err != nil || extrapolated.ExtraUS != b.ExtraUS {
		t.Fatal("profile envelope changed the formula", extrapolated, err)
	}
	requireProfileWarning(t, c.profileWarnings.snapshot(), "restore_service", "candidates")
}

func TestRestoreServiceFlowsThroughEventsAndZeroRetainsBehavior(t *testing.T) {
	c := budgetLabConfig()
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.RestoreServiceCost = serviceProfile(c)
	zero, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Events, zero.Events) || !reflect.DeepEqual(base.FirstTokenUS, zero.FirstTokenUS) || !reflect.DeepEqual(base.BatchShapes, zero.BatchShapes) {
		t.Fatal("zero service changed execution")
	}
	c.DecisionPolicy.RestoreServiceCost.RatesUS["compute_steps"] = 2.5
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	var charged bool
	for _, record := range paid.PolicyDecisions {
		w, err := RestoreDecisionWork(c, record.View)
		if err != nil {
			t.Fatal(err)
		}
		want := int64(math.Ceil(float64(w["compute_steps"]) * 2.5))
		if record.Service == nil || record.Service.ExtraUS != want || record.Feedback.AtUS != record.View.NowUS+want {
			t.Fatal("cost did not execute on the decision service boundary")
		}
		charged = charged || want > 0
	}
	if !charged || paid.FirstTokenUS["warm"] <= base.FirstTokenUS["warm"] {
		t.Fatalf("real first token not delayed: %v %v", base.FirstTokenUS, paid.FirstTokenUS)
	}
	c.DecisionPolicy.ExtraCost = &sim.LinearDecisionCost{Provenance: "conflicting cost"}
	if _, err = Run(c); err == nil {
		t.Fatal("double charging allowed")
	}
}

func TestRestoreServiceProfileValidation(t *testing.T) {
	for _, mutate := range []func(*RestoreServiceCostConfig){
		func(p *RestoreServiceCostConfig) { p.RatesUS["fixed"] = math.NaN() },
		func(p *RestoreServiceCostConfig) { p.RatesUS["fixed"] = -1 },
		func(p *RestoreServiceCostConfig) { p.RatesUS["extra"] = 1 },
		func(p *RestoreServiceCostConfig) { p.Coverage = "" },
		func(p *RestoreServiceCostConfig) { p.Shape.BlockTokens = 0 },
	} {
		c := budgetLabConfig()
		c.DecisionPolicy.RestoreServiceCost = serviceProfile(c)
		mutate(c.DecisionPolicy.RestoreServiceCost)
		if _, err := newRestoreServiceCost(c); err == nil {
			t.Fatal("invalid profile accepted")
		}
	}
}
