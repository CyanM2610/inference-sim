package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestWaitExecutionMappingSeparatesRejectedLoadAndDeferredLookup(t *testing.T) {
	w := sim.BatchExecutionWork{TokenChecksKnown: true,
		TokenChecks: []sim.ExecutionTokenCheck{
			{Request: "held", Phase: "running", Held: true, AllocationStart: 0},
			{Request: "rejected-load", Phase: "waiting", Tokens: 16, AllocationStart: 0},
			{Request: "pending-store", Phase: "waiting", Tokens: 16, AllocationStart: 1},
			{Request: "compute", Phase: "waiting", Tokens: 16, AllocationStart: 2}},
		Allocations: []sim.ExecutionAllocation{
			{Request: "rejected-load", Failure: &sim.AllocationFailure{Request: "rejected-load", Kind: "capacity", RestoreLookupKnown: true, RestoreCandidateBlocks: 2}},
			{Request: "pending-store", Failure: &sim.AllocationFailure{Request: "pending-store", Kind: "wait", Reason: "restore_lookup_pending"}},
			{Request: "compute", Granted: true}},
		Prefixes: []sim.ExecutionPrefix{{Request: "rejected-load"}, {Request: "pending-store"}, {Request: "compute"}}}
	before := w.Allocations[0].Failure.RestoreCandidateBlocks
	counts, err := WaitExecutionCounts(w, 0)
	if err != nil || !reflect.DeepEqual(counts, []int64{1, 2, 1, 0, 3, 2}) || w.Allocations[0].Failure.RestoreCandidateBlocks != before {
		t.Fatal("lookup/allocator/reserve mapping is incorrect", counts, err)
	}
	w.Allocations[0].Failure.RestoreLookupKnown = false
	if _, err := WaitExecutionCounts(w, 0); err == nil {
		t.Fatal("unknown failed LOAD was priced as compute")
	}
	w.Allocations[0].Failure.RestoreLookupKnown = true
	w.Allocations[1].Failure.Reason = "restore_in_flight"
	if _, err := WaitExecutionCounts(w, 0); err == nil {
		t.Fatal("uncovered wait was silently priced")
	}
}

func TestWaitExecutionServiceUsesRuntimeAndZeroPreservesEvents(t *testing.T) {
	c := restoreLabConfig(nil)
	c.PrefillChunk = 16
	c.DecisionPolicy.PrefillWaits = true
	c.DecisionPolicy.ControlSteps = true
	c.DecisionPolicy.ExecutionWork = true
	c.DecisionPolicy.PrefillWaitProbe = &PrefillWaitProbeConfig{DurationUS: 1000, Requests: []string{"warm"}}
	for i := range c.Requests {
		c.Requests[i].MaxOutputTokens = 1
	}
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.WaitExecutionCost = &WaitExecutionCostConfig{Family: "basic", RatesUS: []int64{0, 0, 0, 0, 0, 0},
		MaxCounts: []int64{4, 0, 0, 4, 4}, MaxInputTokens: 64, MaxOutputTokens: 1, HBMBlocks: 4, Shape: serviceShape(c), Provenance: "synthetic wait execution test"}
	zero, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Events, zero.Events) || !reflect.DeepEqual(base.FirstTokenUS, zero.FirstTokenUS) || !reflect.DeepEqual(base.HBM, zero.HBM) {
		t.Fatal("zero execution cost changed behavior")
	}
	c.DecisionPolicy.WaitExecutionCost.RatesUS = []int64{5, 0, 0, 0, 2, 3}
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if paid.FirstTokenUS["warm"] <= zero.FirstTokenUS["warm"] || paid.WaitExecutionCostCoverage == "" || len(paid.FinishedUS) != 3 {
		t.Fatal("host execution service was not paid on the real path")
	}
	for _, record := range paid.PolicyDecisions {
		work := sim.BatchExecutionWork{TokenChecksKnown: true}
		if record.ExecutionWork != nil {
			work = *record.ExecutionWork
		}
		counts, err := WaitExecutionCounts(work, int64(len(record.Feedback.Preemptions)))
		if err != nil || record.ExecutionService == nil || record.ExecutionService.ExtraUS != 5+2*counts[4]+3*counts[5] {
			t.Fatal("observed service does not price actual operations", counts, err)
		}
	}
	c.DecisionPolicy.WaitExecutionCost.MaxCounts[4] = 0
	model, err := newWaitExecutionCost(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.EstimateExecution(sim.BatchExecutionWork{TokenChecksKnown: true, TokenChecks: []sim.ExecutionTokenCheck{{Phase: "running", Held: true}}}, 0); err == nil {
		t.Fatal("unmeasured reserve count accepted")
	}
	c.DecisionPolicy.WaitExecutionCost.Family = "capacity"
	if err := c.Validate(); err == nil {
		t.Fatal("mismatched wrapper family accepted")
	}
}
