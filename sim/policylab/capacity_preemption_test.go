package policylab

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func pressureBudgetConfig() Config {
	c := budgetLabConfig()
	c.DecisionPolicy.BudgetRestore.Mode = "budget_pressure"
	c.DecisionPolicy.PrefillPreemption, c.DecisionPolicy.CapacityPreemption = true, true
	c.DecisionPolicy.ControlSteps = true
	c.DecodeCapacityReservation = true
	c.Instances[0].HBMBlocks, c.MaxSequences, c.PrefillChunk = 10, 4, 16
	c.Requests = nil
	for i := 0; i < 24; i++ {
		slo := int64(5000 + (i%3)*500)
		if i >= 4 && i < 8 {
			slo = 500 // a later urgent burst must displace a looser resident
		}
		tokens := make([]sim.TokenID, 32+16*(i%3))
		for j := range tokens {
			tokens[j] = sim.TokenID(1000*(i%4) + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i * 20), Input: tokens, Output: []sim.TokenID{9, 10, 11, 12}, MaxOutputTokens: 4, TTFTSLOUS: slo})
	}
	return c
}

func TestCapacityBudgetCompletesAndReplaysActualPressure(t *testing.T) {
	c := pressureBudgetConfig()
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.FirstTokenUS) != 24 || len(r.FinishedUS) != 24 || r.DecisionAdditionalCostCoverage != "initial_budget_and_capacity_preemption_not_independently_calibrated" {
		t.Fatal("incomplete workload or missing new cost scope")
	}
	p, _ := sim.NewBudgetPressurePolicy(1, 0)
	actions, waits := 0, 0
	initial := map[string]int64{}
	for _, record := range r.PolicyDecisions {
		if !reflect.DeepEqual(p.Decide(record.View), record.Plan) {
			t.Fatal("actual pressure plan differs from sequential replay")
		}
		for _, b := range p.RequestBudgets(record.View.NowUS) {
			if !b.Known {
				continue
			}
			if old, ok := initial[b.Request]; ok && old != b.InitialUS {
				t.Fatal("actual preemption changed initial budget")
			}
			initial[b.Request] = b.InitialUS
		}
		victims := map[string]bool{}
		for _, outcome := range record.Feedback.Capacity {
			if outcome.Status == "waiting" {
				waits++
			}
			if outcome.Victim != "" {
				actions++
				if outcome.Failure.Kind != "capacity" || victims[outcome.Victim] {
					t.Fatal("victim without actual pressure or repeated same-step victim")
				}
				victims[outcome.Victim] = true
			}
		}
		for _, grant := range record.Feedback.Grants {
			if victims[grant.Request] {
				t.Fatal("yielded request computed in the same step")
			}
		}
	}
	if actions == 0 || waits == 0 || len(initial) != 24 {
		t.Fatal("pressure test failed to exercise actual actions/waits/history", actions, waits, len(initial))
	}
	for _, pool := range r.Pools {
		if pool["reserved"] != 0 || pool["read_pins"] != 0 {
			t.Fatal("CPU lease leaked")
		}
	}
	for _, hbm := range r.HBM {
		if hbm["active_or_pinned"] != 0 || hbm["decode_reserved_blocks"] != 0 || hbm["decode_capacity_requests"] != 0 {
			t.Fatal("HBM lease leaked")
		}
	}
}

func TestCapacityBudgetConfigurationRejectsUnsupportedMechanisms(t *testing.T) {
	for _, modify := range []func(*Config){
		func(c *Config) { c.DecisionPolicy.CapacityPreemption = false },
		func(c *Config) { c.DecisionPolicy.PrefillPreemption = false },
		func(c *Config) { c.BatchPrefixReuse = true },
		func(c *Config) { c.DecodeCapacityReservation = false },
		func(c *Config) { c.Requests[0].MaxOutputTokens = 0 },
	} {
		c := pressureBudgetConfig()
		modify(&c)
		if c.Validate() == nil {
			t.Fatal("unsupported pressure configuration accepted")
		}
	}
}

func TestCapacityBudgetDoesNotPingPongWhenOnlyOneRequestFits(t *testing.T) {
	for _, mode := range []string{"budget_pressure", "budget_pressure_fcfs"} {
		c := pressureBudgetConfig()
		c.DecisionPolicy.BudgetRestore.Mode = mode
		c.Requests = nil
		for i := 0; i < 3; i++ {
			tokens := make([]sim.TokenID, 64)
			for j := range tokens {
				tokens[j] = sim.TokenID(i*1000 + j)
			}
			c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i * 20), Input: tokens,
				Output: make([]sim.TokenID, 20), MaxOutputTokens: 20, TTFTSLOUS: 100000})
		}
		r, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		if len(r.FinishedUS) != 3 || len(r.FirstTokenUS) != 3 || len(r.PolicyDecisions) > 200 {
			t.Fatal("lower-priority admissions repeatedly displaced running work", mode)
		}
	}
}
