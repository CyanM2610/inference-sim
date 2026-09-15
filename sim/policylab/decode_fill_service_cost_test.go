package policylab

import (
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func decodeFillCostFixture() Config {
	c := decodeFillConfig()
	c.DecisionPolicy.ExtraCost = nil
	c.DecisionPolicy.ControlSteps = true
	syntheticWorkerCosts(&c)
	c.EnginePhases.WorkerMetadata.SingleTokenWithNew = &c.EnginePhases.WorkerMetadata.SingleToken
	curve := func(n int) DecodeFillServiceCurve {
		x := DecodeFillServiceCurve{CoefficientsNS: make([]int64, n), MaxFeatures: make([]int64, n)}
		for i := range x.MaxFeatures {
			x.MaxFeatures[i] = 1000
		}
		x.MaxFeatures[0] = 1
		return x
	}
	c.DecisionPolicy.DecodeFillServiceCost = &DecodeFillServiceCostConfig{
		Policy: "decode_fill", Prepare: curve(8), Execution: curve(6), Shape: serviceShape(c),
		HBMBlocks: c.Instances[0].HBMBlocks, PoolCapacityBlocks: c.Pools[0].CapacityBlocks,
		BlockBytes: c.EnginePhases.BlockBytes, MaxInputTokens: 497, MaxOutputTokens: 64,
		MaxPromotionBlocks: 4, Provenance: "synthetic CPU phase test", Coverage: "functional only",
	}
	return c
}

func TestDecodeFillServiceZeroPreservesExecutionAndPaidPhasesPrecedeConsumption(t *testing.T) {
	c := decodeFillCostFixture()
	p := c.DecisionPolicy.DecodeFillServiceCost
	c.DecisionPolicy.DecodeFillServiceCost = nil
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.DecodeFillServiceCost = p
	zero, err := Run(c)
	if err != nil || !reflect.DeepEqual(base.Events, zero.Events) || !reflect.DeepEqual(base.FirstTokenUS, zero.FirstTokenUS) || !reflect.DeepEqual(base.FinishedUS, zero.FinishedUS) {
		t.Fatal("zero profile changed physical execution", err)
	}
	p.Prepare.CoefficientsNS[0] = 1200
	p.Execution.CoefficientsNS[0] = 2800
	p.Execution.CoefficientsNS[4] = 1300 // actual new workers, not requested token caps
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(paid.FinishedUS) != 6 || paid.FirstTokenUS["seed"] <= zero.FirstTokenUS["seed"] {
		t.Fatal("CPU service did not delay actual work")
	}
	var promotions, freshRounds, residentRounds int
	for _, d := range paid.PolicyDecisions {
		_, f, err := DecodeFillServiceFeatures(d.View, d.Plan, d.Feedback.Grants)
		if err != nil {
			t.Fatal(err)
		}
		want := (2800 + 1300*f[4] + 999) / 1000
		if d.Service == nil || d.ExecutionService == nil || d.Service.ExtraUS != 2 || d.ExecutionService.ExtraUS != want ||
			d.ExecutionService.StartUS != d.Service.EndUS || d.Feedback.AtUS != d.ExecutionService.EndUS {
			t.Fatal("wrong CPU boundary, rounding, or premature feedback", d)
		}
		if f[4] > 0 {
			freshRounds++
		} else if len(d.Feedback.Grants) > 0 {
			residentRounds++
		}
		for _, o := range d.Feedback.Promotions {
			if o.StartedBlocks == 0 {
				continue
			}
			promotions++
			for _, e := range paid.Events {
				if e.Request == o.Request && e.Reason == "promotion" && e.Name == "hbm_reserve" && e.Time < d.Service.EndUS {
					t.Fatal("promotion reservation preceded prepare completion", e)
				}
			}
		}
	}
	if promotions == 0 || freshRounds == 0 || residentRounds == 0 {
		t.Fatal("missing distinct service paths", promotions, freshRounds, residentRounds)
	}
	if paid.HBM["instance_0"]["active_or_pinned"] != 0 || paid.Pools["dram"]["reserved"] != 0 || paid.Pools["dram"]["read_pins"] != 0 {
		t.Fatal("CPU service leaked ownership")
	}
}

func TestDecodeFillServiceUsesDispatchedWorkersNotProposedCaps(t *testing.T) {
	v := sim.DecisionView{BlockTokens: 16, Waiting: []sim.DecisionRequest{{ID: "new", InputTokens: 32}},
		Running: []sim.DecisionRequest{{ID: "old", InputTokens: 16, ComputedTokens: 16, EmittedTokens: 1}},
		KV:      sim.DecisionKVState{WorkerStateKnown: true, Requests: []sim.DecisionKVRequest{{ID: "new"}, {ID: "old", WorkerResident: true}}}}
	p := sim.DecisionPlan{TokenCaps: []sim.DecisionTokenCap{{Request: "new", Tokens: 16}, {Request: "old", Tokens: 1}}}
	_, f, err := DecodeFillServiceFeatures(v, p, []sim.DecisionGrant{{Request: "old", Tokens: 1}})
	if err != nil || f[3] != 1 || f[4] != 0 {
		t.Fatal("unchosen candidate charged as new worker", f, err)
	}
	_, f, err = DecodeFillServiceFeatures(v, p, []sim.DecisionGrant{{Request: "new", Tokens: 1}})
	if err != nil || f[4] != 1 {
		t.Fatal("actual new worker missing", f, err)
	}
	for _, grants := range [][]sim.DecisionGrant{{{Request: "absent", Tokens: 1}}, {{Request: "old", Tokens: 0}}, {{Request: "old", Tokens: 1}, {Request: "old", Tokens: 1}}} {
		if _, _, err := DecodeFillServiceFeatures(v, p, grants); err == nil {
			t.Fatal("invalid actual work accepted", grants)
		}
	}
}

func TestDecodeFillServiceRejectsUnmeasuredOrDuplicateProfiles(t *testing.T) {
	for name, change := range map[string]func(*Config){
		"duplicate":            func(c *Config) { c.DecisionPolicy.ExtraCost = &sim.LinearDecisionCost{Provenance: "duplicate"} },
		"wrong policy":         func(c *Config) { c.DecisionPolicy.BalancedBatch.DecodeFill = false },
		"new branch":           func(c *Config) { c.DecisionPolicy.BalancedBatch.ReadyComputeBudget = true },
		"reservation observer": func(c *Config) { c.DecisionPolicy.CapacityReservationView = true },
		"shape":                func(c *Config) { c.DecisionPolicy.DecodeFillServiceCost.Shape.BlockTokens = 0 },
		"request":              func(c *Config) { c.Requests[0].MaxOutputTokens = -1 },
		"negative":             func(c *Config) { c.DecisionPolicy.DecodeFillServiceCost.Execution.CoefficientsNS[0] = -1 },
		"overflow":             func(c *Config) { c.DecisionPolicy.DecodeFillServiceCost.Prepare.CoefficientsNS[0] = math.MaxInt64 },
	} {
		t.Run(name, func(t *testing.T) {
			c := decodeFillCostFixture()
			change(&c)
			if c.Validate() == nil {
				t.Fatal("unsupported model accepted")
			}
		})
	}
	c := decodeFillCostFixture()
	called := false
	if _, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy { called = true; return nil }}); err == nil || called {
		t.Fatal("custom implementation reused calibrated profile")
	}
	m, err := newDecodeFillServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.DecodeFillServiceCost.Prepare.CoefficientsNS[0] = 9999
	if m.profile.Prepare.CoefficientsNS[0] != 0 {
		t.Fatal("caller mutated installed model")
	}
	// Zero fitted rates remain explicitly unvalidated beyond their envelope.
	warnings := &profileWarnings{}
	curve := DecodeFillServiceCurve{CoefficientsNS: []int64{0}, MaxFeatures: []int64{0},
		profileReporter: profileReporter{warnings: warnings, component: "decode_fill_prepare"}}
	if cost, err := priceDecodeFillCurve(curve, []int64{1}); err != nil || cost != 0 {
		t.Fatal("declared formula was replaced", cost, err)
	}
	requireProfileWarning(t, warnings.snapshot(), "decode_fill_prepare", "feature_0")
}
