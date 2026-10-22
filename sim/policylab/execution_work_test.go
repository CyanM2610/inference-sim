package policylab

import (
	"reflect"
	"testing"
)

func TestExecutionWorkPreservesPressureExecutionAndDistinguishesWaits(t *testing.T) {
	c := pressureBudgetConfig()
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.ExecutionWork = true
	observed, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Events, observed.Events) || !reflect.DeepEqual(base.FirstTokenUS, observed.FirstTokenUS) || !reflect.DeepEqual(base.FinishedUS, observed.FinishedUS) || !reflect.DeepEqual(base.HBM, observed.HBM) || !reflect.DeepEqual(base.Pools, observed.Pools) {
		t.Fatal("work observation changed execution or resource ownership")
	}
	if len(base.PolicyDecisions) != len(observed.PolicyDecisions) {
		t.Fatal("decision count changed")
	}
	var attempts, success, waits, failures, queries, noActionCandidates, restoreChecks int
	for i, record := range observed.PolicyDecisions {
		old := base.PolicyDecisions[i]
		if old.ExecutionWork != nil || !reflect.DeepEqual(record.Feedback, old.Feedback) || !reflect.DeepEqual(record.Plan, old.Plan) {
			t.Fatal("observation changed policy behavior")
		}
		if record.ControlStep {
			continue
		}
		if record.ExecutionWork == nil {
			t.Fatal("missing formation work")
		}
		work := record.ExecutionWork
		if !work.TokenChecksKnown {
			t.Fatal("token work remained unknown")
		}
		for index, check := range work.TokenChecks {
			end := len(work.Allocations)
			if index+1 < len(work.TokenChecks) {
				end = work.TokenChecks[index+1].AllocationStart
			}
			if check.AllocationStart < 0 || check.AllocationStart > end || end > len(work.Allocations) {
				t.Fatal("invalid token/allocation association", check, end)
			}
			for _, allocation := range work.Allocations[check.AllocationStart:end] {
				if allocation.Request != check.Request {
					t.Fatal("allocation belongs to another token check", check, allocation)
				}
				if allocation.Failure != nil && allocation.Failure.Reason == "restore_submitted" {
					if check.Phase != "waiting" {
						t.Fatal("restore submission associated with a running check", check)
					}
					restoreChecks++
				}
			}
		}
		queries += len(record.ExecutionWork.Prefixes)
		for _, a := range record.ExecutionWork.Allocations {
			attempts++
			if a.Granted {
				success++
				if a.Failure != nil {
					t.Fatal("stale failure on success")
				}
				continue
			}
			if a.Failure == nil || a.Failure.Request != a.Request {
				t.Fatal("missing actual failure identity")
			}
			switch a.Failure.Kind {
			case "capacity":
				failures++
			case "wait":
				waits++
			default:
				t.Fatal("unknown failure", a)
			}
		}
		if record.Plan.CapacityVictims != nil && len(record.Plan.CapacityVictims.Order) > 0 && len(record.Feedback.Preemptions) == 0 {
			noActionCandidates++
		}
	}
	if attempts == 0 || success == 0 || waits == 0 || failures == 0 || queries == 0 || noActionCandidates == 0 || restoreChecks == 0 {
		t.Fatal("did not cover success, pressure, wait, prefix lookup and unused candidates", attempts, success, waits, failures, queries, noActionCandidates)
	}
}
