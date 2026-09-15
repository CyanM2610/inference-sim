package policylab

import "testing"

func TestResidentPrefillWaitSurvivesActualCapacityPreemption(t *testing.T) {
	c := pressureBudgetConfig()
	c.DecisionPolicy.PrefillWaits = true
	c.DecisionPolicy.PrefillWaitProbe = &PrefillWaitProbeConfig{DurationUS: 1000}
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.FinishedUS) != 24 || len(r.FirstTokenUS) != 24 {
		t.Fatal("held resident failed to complete after real capacity pressure", r.FinishedUS)
	}
	waits := map[string]int64{}
	preempted := map[string]int64{}
	readmitted := map[string]bool{}
	pressure := 0
	for _, record := range r.PolicyDecisions {
		for _, update := range record.Feedback.WaitUpdates {
			waits[update.Request] = update.UntilUS
		}
		for _, outcome := range record.Feedback.Capacity {
			if outcome.Victim != "" && waits[outcome.Victim] > record.Feedback.AtUS {
				if outcome.Failure.Kind != "capacity" {
					t.Fatal("wait treated as capacity pressure", outcome)
				}
				preempted[outcome.Victim] = waits[outcome.Victim]
				pressure++
			}
		}
		for _, q := range record.View.Waiting {
			if until, ok := preempted[q.ID]; ok && record.View.NowUS < until && q.WaitUntilUS != until {
				t.Fatal("capacity requeue lost resident deadline", q, until)
			}
		}
		for _, grant := range record.Feedback.Grants {
			if until, ok := preempted[grant.Request]; ok {
				if record.Feedback.AtUS < until {
					t.Fatal("victim ran before preserved expiry", grant, record.Feedback.AtUS, until)
				}
				readmitted[grant.Request] = true
			}
		}
	}
	if pressure == 0 || len(preempted) != len(readmitted) {
		t.Fatal("did not exercise paused victim and actual reentry", pressure, preempted, readmitted)
	}
	for _, pool := range r.Pools {
		if pool["reserved"] != 0 || pool["read_pins"] != 0 {
			t.Fatal("CPU lease leaked", pool)
		}
	}
	for _, hbm := range r.HBM {
		if hbm["active_or_pinned"] != 0 || hbm["decode_reserved_blocks"] != 0 || hbm["decode_capacity_requests"] != 0 {
			t.Fatal("HBM ownership/reservation leaked", hbm)
		}
	}
	t.Logf("completed=%d decisions=%d held_capacity_preemptions=%d held_victims=%d reentered=%d", len(r.FinishedUS), len(r.PolicyDecisions), pressure, len(preempted), len(readmitted))
}
