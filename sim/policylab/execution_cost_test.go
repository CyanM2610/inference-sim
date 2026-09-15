package policylab

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

func TestCapacityExecutionCostUsesActualReservationAndAdoptionSemantics(t *testing.T) {
	c := pressureBudgetConfig()
	// This unit test uses the synthetic fixture's own declared geometry.
	c.DecisionPolicy.ExecutionCost = &CapacityExecutionCostConfig{RatesUS: []int64{10, 55, 8, 25, 6}, ActionUS: 21, MaxCounts: []int64{4, 1, 1, 2}, Shape: serviceShape(c), Provenance: "synthetic operation rates"}
	c.DecisionPolicy.ExecutionCost.MaxInputTokens, c.DecisionPolicy.ExecutionCost.MaxOutputTokens, c.DecisionPolicy.ExecutionCost.HBMBlocks = 64, 4, c.Instances[0].HBMBlocks
	m, err := newCapacityExecutionCost(c)
	if err != nil {
		t.Fatal(err)
	}
	w := sim.BatchExecutionWork{Allocations: []sim.ExecutionAllocation{
		{Request: "load", Failure: &sim.AllocationFailure{Request: "load", Kind: "wait", Reason: "restore_submitted"}},
		{Request: "blocked", Failure: &sim.AllocationFailure{Request: "blocked", Kind: "capacity"}},
		{Request: "ready", Granted: true},
	}, Prefixes: []sim.ExecutionPrefix{{Request: "load"}, {Request: "blocked"}, {Request: "ready", AdoptingRestore: true}}}
	x, err := m.EstimateExecution(w, 1)
	if err != nil || x.ExtraUS != 10+3*55+8+25+2*6+21 {
		t.Fatal("wrong actual operation mapping", x, err)
	}
	x, err = m.EstimateExecution(sim.BatchExecutionWork{}, 0)
	if err != nil || x.ExtraUS != 10 {
		t.Fatal("empty control step priced as work or free", x, err)
	}
	w.Allocations[0].Failure.Reason = "restore_in_flight"
	if _, err = m.EstimateExecution(w, 1); err == nil {
		t.Fatal("unsupported wait mapping silently extrapolated")
	}
	if _, err = m.EstimateExecution(sim.BatchExecutionWork{}, 2); err == nil {
		t.Fatal("unsupported actual victim count accepted")
	}
}
