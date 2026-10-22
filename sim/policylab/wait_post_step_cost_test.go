package policylab

import (
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestWaitPostStepZeroCompatibilityAndPositiveCosts(t *testing.T) {
	c := restoreLabConfig(nil)
	c.PrefillChunk = 16
	c.DecisionPolicy.PrefillWaits, c.DecisionPolicy.ControlSteps = true, true
	c.DecisionPolicy.PrefillWaitProbe = &PrefillWaitProbeConfig{DurationUS: 1000, Requests: []string{"warm"}}
	for i := range c.Requests {
		c.Requests[i].MaxOutputTokens = 1
	}
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.WaitPostStepCost = &WaitPostStepCostConfig{Family: "basic", IdleRatesUS: []int64{0, 0, 0}, MaxIdleCounts: []int64{1, 4, 4},
		MaxInputTokens: 64, MaxOutputTokens: 1, HBMBlocks: 4, Shape: serviceShape(c), Provenance: "synthetic post-step services"}
	zero, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Events, zero.Events) || !reflect.DeepEqual(base.FirstTokenUS, zero.FirstTokenUS) || !reflect.DeepEqual(base.HBM, zero.HBM) {
		t.Fatal("zero post-step service changed execution")
	}
	p := c.DecisionPolicy.WaitPostStepCost
	p.BookkeepingUS, p.OutputAccountingUS, p.IdleRatesUS = 1, 2, []int64{3, 1, 2}
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(paid.FinishedUS) != 3 || len(paid.PostStepServices) == 0 || paid.PostStepCostCoverage == "" {
		t.Fatal("post-step services did not run or drain")
	}
	for _, r := range paid.PostStepServices {
		want := int64(4)
		if r.Work.Stage == "idle_check" {
			want = 3 + int64(len(r.Work.Requests)) + 2*int64(len(r.Work.Held))
		}
		if r.Service.ExtraUS != want || r.Service.EndUS-r.Service.StartUS != want {
			t.Fatal("post-step amount differs from actual work", r)
		}
	}
	p.PreStepRatesUS = []int64{0, 0, 0}
	loopZero, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(paid.Events, loopZero.Events) || !reflect.DeepEqual(paid.FirstTokenUS, loopZero.FirstTokenUS) {
		t.Fatal("zero driver prefix changed existing execution")
	}
	p.PreStepRatesUS = []int64{2, 1, 3}
	loopPaid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	prefixes, exits := 0, 0
	for _, r := range loopPaid.PostStepServices {
		if r.Work.Stage == "before_step" {
			prefixes++
			if r.Service.ExtraUS != 6 {
				t.Fatal("wrong driver prefix fee", r)
			}
		} else if r.Work.Stage == "loop_exit" {
			exits++
			if r.Service.ExtraUS != 2 {
				t.Fatal("final condition charged a full loop", r)
			}
		}
	}
	if prefixes == 0 || exits != 1 || len(loopPaid.FinishedUS) != len(c.Requests) {
		t.Fatal("driver prefix or exit omitted", prefixes, exits)
	}
	m, err := newWaitPostStepCost(c)
	if err != nil {
		t.Fatal(err)
	}
	p.IdleRatesUS[0] = 999
	if fee, err := m.EstimatePostStep(sim.PostStepWork{Stage: "idle_check"}); err != nil || fee.ExtraUS != 3 {
		t.Fatal("installed profile aliases config", fee, err)
	}
	if _, err := m.EstimatePostStep(sim.PostStepWork{Stage: "idle_check", Held: []string{"missing"}}); err == nil {
		t.Fatal("missing held request accepted")
	}
	m.config.BookkeepingUS = math.MaxInt64
	if _, err := m.EstimatePostStep(sim.PostStepWork{Stage: "after_output"}); err == nil {
		t.Fatal("post-step overflow accepted")
	}
	p.PreStepRatesUS[0] = 999
	if fee, err := m.EstimatePostStep(sim.PostStepWork{Stage: "before_step"}); err != nil || fee.ExtraUS != 6 {
		t.Fatal("loop coefficients alias config", fee, err)
	}
	p.PreStepRatesUS = []int64{}
	if _, err := newWaitPostStepCost(c); err == nil {
		t.Fatal("explicit incomplete loop coefficients accepted")
	}
}
