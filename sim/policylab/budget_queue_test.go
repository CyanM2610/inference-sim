package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type boundedBudgetQueue struct {
	*sim.BudgetQueuePolicy
	t     *testing.T
	steps int
}

func (p *boundedBudgetQueue) Decide(v sim.DecisionView) sim.DecisionPlan {
	p.steps++
	if p.steps > 2000 {
		p.t.Fatalf("budget queues exceeded progress bound: %+v", v)
	}
	return p.BudgetQueuePolicy.Decide(v)
}

func TestBudgetQueuesComposeThroughRuntimeAndReplayActualFeedback(t *testing.T) {
	for _, pressure := range []bool{false, true} {
		name := "fractional"
		if pressure {
			name = "capacity"
		}
		t.Run(name, func(t *testing.T) {
			c := pressureBudgetConfig()
			c.DecisionPolicy.BudgetRestore = &BudgetRestoreConfig{Mode: "budget_queues", Guardband: 1, PrimaryWeight: 3, BestEffortWeight: 1}
			if !pressure {
				c.Instances[0].HBMBlocks, c.Pools[0].CapacityBlocks = 20, 32
				c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 2, 7, 7
				c.Requests = nil
				for i, id := range []string{"primary", "best"} {
					input := make([]sim.TokenID, 64)
					for j := range input {
						input[j] = sim.TokenID(1000*i + j)
					}
					slo := int64(1000000000)
					if i == 1 {
						slo = 1
					}
					c.Requests = append(c.Requests, RequestConfig{ID: id, Input: input, Output: []sim.TokenID{9, 10, 11}, MaxOutputTokens: 3, TTFTSLOUS: slo})
				}
			}
			var used *boundedBudgetQueue
			injected, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy {
				p, e := sim.NewBudgetQueuePolicy(1, 0, 3, 1)
				if e != nil {
					t.Fatal(e)
				}
				used = &boundedBudgetQueue{BudgetQueuePolicy: p, t: t}
				return used
			}})
			if err != nil {
				t.Fatal(err)
			}
			configured, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(configured.Events, injected.Events) || !reflect.DeepEqual(configured.FirstTokenUS, injected.FirstTokenUS) {
				t.Fatal("configuration and public factory execute differently")
			}
			if len(configured.FinishedUS) != len(c.Requests) || len(configured.FirstTokenUS) != len(c.Requests) ||
				configured.DecisionAdditionalCostCoverage != "budget_queues_actual_grant_accounting_not_independently_calibrated" {
				t.Fatal("missing completion or new cost scope")
			}
			replay, _ := sim.NewBudgetQueuePolicy(1, 0, 3, 1)
			var actualPrefill int64
			preemptions, emptyGrants := 0, 0
			for _, d := range configured.PolicyDecisions {
				if !reflect.DeepEqual(replay.Decide(d.View), d.Plan) {
					t.Fatal("retained plan replay differs")
				}
				replay.Observe(d.Feedback)
				before := map[string]sim.DecisionRequest{}
				for _, r := range append(append([]sim.DecisionRequest(nil), d.View.Waiting...), d.View.Running...) {
					before[r.ID] = r
				}
				for _, g := range d.Feedback.Grants {
					r := before[g.Request]
					if r.EmittedTokens == 0 && r.ComputedTokens < r.InputTokens {
						actualPrefill += g.Tokens
					}
				}
				preemptions += len(d.Feedback.Preemptions)
				if len(d.Feedback.Grants) == 0 {
					emptyGrants++
				}
			}
			state := used.ServiceState()
			if replay.ServiceState() != state || state.PrimaryTokens+state.BestEffortTokens != actualPrefill {
				t.Fatal("accounting differs from actual grants", state, actualPrefill)
			}
			if !pressure {
				if state.PrimaryTokens != 64 || state.BestEffortTokens != 64 || configured.FirstTokenUS["primary"] >= configured.FirstTokenUS["best"] {
					t.Fatal("shares did not change actual request service", state, configured.FirstTokenUS)
				}
				c.DecisionPolicy.BudgetRestore = &BudgetRestoreConfig{Mode: "budget_pressure", Guardband: 1}
				baseline, e := Run(c)
				if e != nil {
					t.Fatal(e)
				}
				if baseline.FirstTokenUS["best"] >= baseline.FirstTokenUS["primary"] {
					t.Fatal("single-queue counterexample did not differ")
				}
			} else if preemptions == 0 || emptyGrants == 0 {
				t.Fatal("pressure path not exercised", preemptions, emptyGrants)
			}
			for _, pool := range configured.Pools {
				if pool["reserved"] != 0 || pool["read_pins"] != 0 {
					t.Fatal("pool lease leaked", pool)
				}
			}
			for _, hbm := range configured.HBM {
				if hbm["active_or_pinned"] != 0 || hbm["decode_reserved_blocks"] != 0 || hbm["decode_capacity_requests"] != 0 {
					t.Fatal("HBM lease leaked", hbm)
				}
			}
		})
	}
}

func TestBudgetQueueConfigurationFailsBeforeEventExecution(t *testing.T) {
	for _, mutate := range []func(*Config){
		func(c *Config) { c.DecisionPolicy.CapacityPreemption = false },
		func(c *Config) { c.DecisionPolicy.BudgetRestore.PrimaryWeight = -1 },
		func(c *Config) {
			c.DecisionPolicy.BudgetRestore.PrimaryWeight = 0
			c.DecisionPolicy.BudgetRestore.BestEffortWeight = 1
		},
		func(c *Config) {
			c.DecisionPolicy.BudgetRestore.Mode = "initial_budget"
			c.DecisionPolicy.BudgetRestore.PrimaryWeight = 3
		},
	} {
		c := pressureBudgetConfig()
		c.DecisionPolicy.BudgetRestore.Mode = "budget_queues"
		mutate(&c)
		if c.Validate() == nil {
			t.Fatal("invalid budget queues configuration accepted")
		}
	}
}
