package sim

import (
	"reflect"
	"testing"
)

func TestBudgetPressureUsesRetainedBudgetAndProtectsUnknownDecode(t *testing.T) {
	v := budgetView()
	v.Capabilities.PrefillPreemption, v.Capabilities.CapacityPreemption = true, true
	p, _ := NewBudgetPressurePolicy(1, 0)
	p.Decide(v)
	fifo, _ := NewBudgetPressureFIFOPolicy(1, 0)
	fifo.Decide(v)
	v.Running, v.Waiting = v.Waiting, nil
	v.Estimates.Requests = nil
	for i := range v.Running {
		v.Running[i].State, v.Running[i].ComputedTokens, v.Running[i].PrefillPreemptible = StateRunning, 2, true
	}
	v.Running[2].ComputedTokens = v.Running[2].InputTokens
	v.Running[2].PrefillPreemptible = false
	v.Running = append(v.Running, DecisionRequest{ID: "unseen", State: StateRunning, InputTokens: 8, ComputedTokens: 2, TTFTTargetUS: 1000, PrefillPreemptible: true})
	v.NowUS = 20
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if len(plan.Preemptions) != 0 || !reflect.DeepEqual(plan.CapacityVictims.Order, []string{"wide", "mid"}) {
		t.Fatal("not conditional maximum retained budget", plan)
	}
	if !reflect.DeepEqual(fifo.Decide(v).CapacityVictims.Order, []string{"mid", "wide"}) {
		t.Fatal("FCFS ablation did not retain running-tail victim order")
	}
	copy := cloneDecisionPlan(plan)
	copy.CapacityVictims.Order[0] = "unseen"
	if plan.CapacityVictims.Order[0] != "wide" {
		t.Fatal("conditional order aliases policy data")
	}
}

func TestCapacityPlanRejectsInvalidVictimsBeforeMutation(t *testing.T) {
	v := budgetView()
	v.Capabilities.PrefillPreemption, v.Capabilities.CapacityPreemption = true, true
	v.Running = []DecisionRequest{{ID: "a", State: StateRunning, InputTokens: 8, ComputedTokens: 2, PrefillPreemptible: true},
		{ID: "decode", State: StateRunning, InputTokens: 2, ComputedTokens: 4}}
	for _, order := range [][]string{{"a", "a"}, {"decode"}, {"future"}, {"wide"}} {
		if err := ValidateDecision(v, DecisionPlan{Version: v.Version, CapacityVictims: &DecisionCapacityVictims{Order: order}}); err == nil {
			t.Fatal("invalid victim accepted", order)
		}
	}
	plan := DecisionPlan{Version: v.Version, CapacityVictims: &DecisionCapacityVictims{Order: []string{"a"}}}
	plan.Preemptions = []string{"a"}
	if ValidateDecision(v, plan) == nil {
		t.Fatal("unconditional and conditional actions can race")
	}
	plan.Preemptions = nil
	v.Capabilities.CapacityPreemption = false
	if ValidateDecision(v, plan) == nil {
		t.Fatal("unsupported action accepted")
	}
}
