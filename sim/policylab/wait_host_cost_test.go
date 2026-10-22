package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func waitHostFixture(c Config) *WaitHostServiceCostConfig {
	registration := int64(20)
	return &WaitHostServiceCostConfig{Family: "capacity", CompletionRatesUS: []int64{3, 9, 4}, MaxCompletionCounts: []int64{1, 4, 4},
		RegistrationUS: &registration, MaxInputTokens: 64, MaxOutputTokens: 4, HBMBlocks: c.Instances[0].HBMBlocks, Shape: serviceShape(c), Provenance: "synthetic wait host boundary test"}
}

func TestWaitHostRegistrationReplacesEnqueueAndValidatesCompletion(t *testing.T) {
	c := pressureBudgetConfig()
	c.DecisionPolicy.PrefillWaits = true
	c.DecisionPolicy.WaitHostServiceCost = waitHostFixture(c)
	c.BatchCost.EnqueueUS = 277
	if _, err := newWaitHostServiceCost(c); err == nil {
		t.Fatal("complete registration added on top of enqueue")
	}
	*c.DecisionPolicy.WaitHostServiceCost.RegistrationUS = 0
	if _, err := newWaitHostServiceCost(c); err == nil {
		t.Fatal("zero complete registration did not require normalized enqueue")
	}
	c.BatchCost.EnqueueUS = 0
	*c.DecisionPolicy.WaitHostServiceCost.RegistrationUS = 20
	m, err := newWaitHostServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	*c.DecisionPolicy.WaitHostServiceCost.RegistrationUS = 1000
	w := sim.HostServiceWork{Stage: "registration", Requests: []string{"a"}}
	if cost, err := m.EstimateHostService(w); err != nil || cost.ExtraUS != 20 {
		t.Fatal("registration profile was not detached", cost, err)
	}
	w = sim.HostServiceWork{Stage: "completion", Requests: []string{"a", "b"}, Finishing: []string{"b"}}
	if cost, err := m.EstimateHostService(w); err != nil || cost.ExtraUS != 25 {
		t.Fatal("wrong completion work charge", cost, err)
	}
	basic, err := WaitCompletionCounts(w, "basic")
	if err != nil || !reflect.DeepEqual(basic, []int64{0, 2, 0}) {
		t.Fatal("basic cleanup confused with capacity wrapper", basic, err)
	}
	for _, bad := range []sim.HostServiceWork{
		{Stage: "completion", Requests: []string{"a"}, Finishing: []string{"b"}},
		{Stage: "completion", Requests: []string{"a"}, Finishing: []string{"a", "a"}},
		{Stage: "completion", Requests: []string{"a", "a"}},
	} {
		if _, err := m.EstimateHostService(bad); err == nil {
			t.Fatal("invalid completion identities accepted")
		}
	}
}

func TestWaitHostZeroPreservesEventsAndPositiveServicesDrain(t *testing.T) {
	c := pressureBudgetConfig()
	c.DecisionPolicy.PrefillWaits = true
	c.DecisionPolicy.PrefillWaitProbe = &PrefillWaitProbeConfig{DurationUS: 1000, Requests: []string{"0"}}
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.WaitHostServiceCost = waitHostFixture(c)
	c.DecisionPolicy.WaitHostServiceCost.CompletionRatesUS = []int64{0, 0, 0}
	*c.DecisionPolicy.WaitHostServiceCost.RegistrationUS = 0
	zero, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Events, zero.Events) || !reflect.DeepEqual(base.FirstTokenUS, zero.FirstTokenUS) || !reflect.DeepEqual(base.HBM, zero.HBM) {
		t.Fatal("zero wait host services changed execution")
	}
	c.DecisionPolicy.WaitHostServiceCost = waitHostFixture(c)
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(paid.FinishedUS) != len(c.Requests) || paid.HostServiceCostCoverage == "" {
		t.Fatal("paid host services did not drain")
	}
	registrations, finishes, completions := 0, 0, 0
	for _, record := range paid.HostServices {
		want := int64(20)
		if record.Work.Stage == "completion" {
			completions++
			finishes += len(record.Work.Finishing)
			want = 3 + 9*int64(len(record.Work.Requests)) + 4*int64(len(record.Work.Finishing))
		} else {
			registrations++
		}
		if record.Service.ExtraUS != want || record.Service.EndUS-record.Service.StartUS != want {
			t.Fatal("work was not charged at host boundary", record)
		}
	}
	if registrations != len(c.Requests) || finishes != len(c.Requests) || completions != len(paid.PolicyDecisions) {
		t.Fatal("host work duplicated or missed", registrations, finishes, completions)
	}
}
