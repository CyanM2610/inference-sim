package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestQueueExecutionCountsRetainRejectedAllocationsAndActualQuotaWork(t *testing.T) {
	v := sim.DecisionView{Version: 4, Waiting: []sim.DecisionRequest{{ID: "blocked"}, {ID: "reader"}}, Running: []sim.DecisionRequest{{ID: "running"}}}
	w := sim.BatchExecutionWork{TokenChecksKnown: true,
		TokenChecks: []sim.ExecutionTokenCheck{{Request: "blocked", Phase: "waiting", AllocationStart: 0}, {Request: "reader", Phase: "waiting", AllocationStart: 1}, {Request: "running", Phase: "running", AllocationStart: 2}},
		Allocations: []sim.ExecutionAllocation{{Request: "blocked", Failure: &sim.AllocationFailure{Request: "blocked", Kind: "capacity", RestoreLookupKnown: true}},
			{Request: "reader", Failure: &sim.AllocationFailure{Request: "reader", Kind: "wait", Reason: "restore_submitted"}}, {Request: "running", Granted: true}},
		Prefixes: []sim.ExecutionPrefix{{Request: "blocked"}, {Request: "reader"}}}
	f := sim.DecisionFeedback{Version: 4, Status: "applied", Grants: []sim.DecisionGrant{{Request: "running", Tokens: 2}}, Capacity: []sim.DecisionCapacityOutcome{{Status: "no_eligible_victim"}}}
	x, err := QueueExecutionCounts(v, w, f, true)
	if err != nil || !reflect.DeepEqual(x["execution"], []int64{1, 3, 1, 0, 2, 2, 1}) || !reflect.DeepEqual(x["feedback"], []int64{1, 3, 1, 1, 0}) {
		t.Fatal("did not distinguish failed allocation, async load and actual grant", x, err)
	}
	y, err := QueueExecutionCounts(v, w, f, false)
	if err != nil || y["execution"][5] != 0 || x["execution"][5] != 2 {
		t.Fatal("pressure inherited quota callback counts", y, err)
	}
	f.Version--
	if _, err := QueueExecutionCounts(v, w, f, true); err == nil {
		t.Fatal("previous feedback reused as current work")
	}
}

func TestQueueDecisionWorkUsesObservedEstimatorChoicesAndCeilingBlocks(t *testing.T) {
	c := pressureBudgetConfig()
	c.DecisionPolicy.ExecutionWork = true
	c.DecisionPolicy.CapacityReservationView = true
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, row := range r.PolicyDecisions {
		counts, err := QueueDecisionCounts(c, row.View, row.Plan)
		if err != nil {
			t.Fatal(err)
		}
		var choices int64
		for _, q := range row.View.Estimates.Requests {
			choices += int64(len(q.Choices))
		}
		if counts["estimate"][2] != choices || counts["decision"][4] != int64(len(row.Plan.TokenCaps)) {
			t.Fatal("wrong decision work", counts)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no runtime decisions checked")
	}
	row := r.PolicyDecisions[0]
	row.View.Waiting[0].InputTokens = 33
	counts, err := QueueDecisionCounts(c, row.View, row.Plan)
	if err != nil || counts["reservation_view"][2] != 3 {
		t.Fatal("partial final block omitted", counts, err)
	}
}
