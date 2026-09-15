package policylab

import (
	"github.com/inference-sim/inference-sim/sim"
	"reflect"
	"testing"
)

func hostProfile(c Config, zero bool) *CapacityHostServiceCostConfig {
	rates := []int64{10, 5, 4}
	registration := int64(8)
	if zero {
		rates = []int64{0, 0, 0}
		registration = 0
	}
	return &CapacityHostServiceCostConfig{CompletionRatesUS: rates, RegistrationUS: &registration, MaxGrantedRequests: 4, MaxFinishingRequests: 4, MaxInputTokens: 64, MaxOutputTokens: 4, HBMBlocks: c.Instances[0].HBMBlocks, Shape: serviceShape(c), Provenance: "synthetic boundary profile"}
}

func TestZeroHostServicesPreserveExecutionAndRecordEveryCompletion(t *testing.T) {
	c := pressureBudgetConfig()
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.HostServiceCost = hostProfile(c, true)
	observed, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Events, observed.Events) || !reflect.DeepEqual(base.FirstTokenUS, observed.FirstTokenUS) || !reflect.DeepEqual(base.FinishedUS, observed.FinishedUS) {
		t.Fatal("zero host cost changed execution")
	}
	var completions, registrations, finished, grants int
	for _, r := range observed.HostServices {
		if r.Service.ExtraUS != 0 || r.Service.StartUS != r.Service.EndUS {
			t.Fatal("zero service advanced time")
		}
		switch r.Work.Stage {
		case "completion":
			completions++
			finished += len(r.Work.Finishing)
			grants += len(r.Work.Requests)
		case "registration":
			registrations++
		default:
			t.Fatal("unknown stage")
		}
	}
	wantGrants := 0
	for _, d := range observed.PolicyDecisions {
		wantGrants += len(d.Feedback.Grants)
	}
	if completions != len(observed.PolicyDecisions) || registrations != len(c.Requests) || finished != len(c.Requests) || grants != wantGrants {
		t.Fatal("host work lost a computation, finish, registration or empty control callback", completions, registrations, finished, grants, wantGrants)
	}
}

func TestHostCostRejectsUncoveredGeometryAndInvalidFinishSets(t *testing.T) {
	c := pressureBudgetConfig()
	c.DecisionPolicy.HostServiceCost = hostProfile(c, false)
	m, err := newCapacityHostServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	x, err := m.EstimateHostService(sim.HostServiceWork{Stage: "completion", Requests: []string{"a", "b"}, Finishing: []string{"b"}})
	if err != nil || x.ExtraUS != 24 {
		t.Fatal(x, err)
	}
	for _, w := range []sim.HostServiceWork{
		{Stage: "completion", Requests: []string{"a"}, Finishing: []string{"b"}},
		{Stage: "completion", Requests: []string{"a"}, Finishing: []string{"a", "a"}},
		{Stage: "completion", Requests: []string{"a", "a"}},
		{Stage: "registration", Requests: []string{"a", "b"}},
	} {
		if _, err := m.EstimateHostService(w); err == nil {
			t.Fatal("invalid work accepted", w)
		}
	}
	c.DecisionPolicy.HostServiceCost.HBMBlocks++
	if _, err := newCapacityHostServiceCost(c); err == nil {
		t.Fatal("different geometry accepted")
	}
}

func TestRegistrationBaseCannotBeChargedTwice(t *testing.T) {
	c := pressureBudgetConfig()
	c.DecisionPolicy.HostServiceCost = hostProfile(c, false)
	c.DecisionPolicy.HostServiceCost.RegistrationBaseUS = 340
	c.BatchCost.EnqueueUS = 277
	if _, err := newCapacityHostServiceCost(c); err == nil {
		t.Fatal("base registration charged both at arrival and host entry")
	}
	c.BatchCost.EnqueueUS = 0
	m, err := newCapacityHostServiceCost(c)
	if err != nil {
		t.Fatal(err)
	}
	x, err := m.EstimateHostService(sim.HostServiceWork{Stage: "registration", Requests: []string{"a"}})
	if err != nil || x.ExtraUS != 348 {
		t.Fatal("base registration missing from host service", x, err)
	}
}
