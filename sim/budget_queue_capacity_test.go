package sim

import "testing"

func reservationQueueView() DecisionView {
	v := budgetQueueView()
	v.Capabilities.CapacityReservations = true
	for i := range v.KV.Requests {
		v.KV.Requests[i].CapacityReservation = &DecisionCapacityReservation{NeededBlocks: 1, FreeBlocks: 4, Fits: true}
	}
	return v
}

func TestBudgetQueuesCapacityBlockBorrowsShareAndReopens(t *testing.T) {
	p, _ := NewBudgetQueuePolicy(1, 0, 4, 1)
	v := reservationQueueView()
	v.KV.Requests[0].CapacityReservation = &DecisionCapacityReservation{NeededBlocks: 5, FreeBlocks: 4}
	plan := p.Decide(v)
	if caps := capsByRequest(plan); caps["primary"] != 0 || caps["best"] != 7 {
		t.Fatalf("blocked queue kept unusable quota: %v", caps)
	}
	p.Observe(DecisionFeedback{Version: v.Version, Status: "applied", Grants: []DecisionGrant{{Request: "best", Tokens: 7}}})
	if p.ServiceState().DeficitUnits != 0 {
		t.Fatal("blocked primary created future debt")
	}
	v.Version++
	v.NowUS++
	v.KV.Requests[0].CapacityReservation = &DecisionCapacityReservation{NeededBlocks: 5, FreeBlocks: 5, Fits: true}
	plan = p.Decide(v)
	if caps := capsByRequest(plan); caps["primary"] != 5 || caps["best"] != 2 {
		t.Fatalf("capacity release did not restore contention: %v", caps)
	}
}

func TestBudgetQueuesActualCapacityFailureEndsFalseContention(t *testing.T) {
	p, _ := NewBudgetQueuePolicy(1, 0, 4, 1)
	v := reservationQueueView()
	p.Decide(v)
	p.Observe(DecisionFeedback{Version: v.Version, Status: "applied", Grants: []DecisionGrant{{Request: "best", Tokens: 2}},
		Capacity: []DecisionCapacityOutcome{{Failure: AllocationFailure{Request: "primary", Kind: "capacity"}, Status: "no_eligible_victim"}}})
	s := p.ServiceState()
	if s.BestEffortTokens != 2 || s.DeficitUnits != 0 || s.ContendedBestTokens != 0 {
		t.Fatalf("failed admission counted as actual contention: %+v", s)
	}
}

func TestCapacityReservationSnapshotCloneDoesNotAlias(t *testing.T) {
	v := reservationQueueView()
	copy := cloneDecisionView(v)
	copy.KV.Requests[0].CapacityReservation.Fits = false
	if !v.KV.Requests[0].CapacityReservation.Fits {
		t.Fatal("cloned capacity state aliases validation input")
	}
}

func TestBudgetQueuesReservationDoesNotDisableAllowedVictims(t *testing.T) {
	p, _ := NewBudgetQueuePolicy(1, 0, 4, 1)
	v := reservationQueueView()
	p.Decide(v)
	p.Observe(DecisionFeedback{Version: v.Version, Status: "applied"})
	v.Version++
	v.Running = []DecisionRequest{v.Waiting[0]}
	v.Running[0].State, v.Running[0].ComputedTokens, v.Running[0].PrefillPreemptible = StateRunning, 16, true
	v.Waiting = v.Waiting[1:]
	v.Estimates.Requests = v.Estimates.Requests[1:]
	v.KV.Requests[1].CapacityReservation = &DecisionCapacityReservation{NeededBlocks: 5, FreeBlocks: 4}
	plan := p.Decide(v)
	if len(plan.Admission.Requests) != 1 || plan.Admission.Requests[0] != "best" || capsByRequest(plan)["best"] == 0 {
		t.Fatal("reservation check hid a request with an authorized victim", plan)
	}
}
