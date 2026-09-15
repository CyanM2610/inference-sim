package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// These use the public injection seam with a deliberately different fallback.
// A factory which is constructed but ignored must fail the actual-work checks.
func TestJointDecisionFactoriesReplaceRestoreAndBatchDecisions(t *testing.T) {
	t.Run("budget restore", func(t *testing.T) {
		c := budgetLabConfig()
		c.Requests[2].TTFTSLOUS = 180
		full := int64(2)
		c.DecisionPolicy.BudgetRestore = nil
		c.DecisionPolicy.RestorePrefixBlocks = &full
		baseline, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		constructed := 0
		injected, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy {
			constructed++
			p, e := sim.NewBudgetRestorePolicy(1, 0)
			if e != nil {
				t.Fatal(e)
			}
			return p
		}})
		if err != nil {
			t.Fatal(err)
		}
		loaded := func(r *Result) int64 {
			var blocks int64
			for _, e := range r.Events {
				if e.Name == "transfer_start" && e.Request == "reader" && e.Destination == "hbm" {
					blocks += e.Bytes / r.BlockBytes
				}
			}
			return blocks
		}
		if constructed != 1 || loaded(baseline) != 2 || loaded(injected) != 1 || len(injected.FinishedUS) != 3 {
			t.Fatalf("factory did not change actual restoration: calls=%d baseline=%d injected=%d", constructed, loaded(baseline), loaded(injected))
		}
		c.DecisionPolicy.RestorePrefixBlocks = nil
		c.DecisionPolicy.BudgetRestore = &BudgetRestoreConfig{Guardband: 1}
		configured, err := Run(c)
		if err != nil || !reflect.DeepEqual(configured.Events, injected.Events) || !reflect.DeepEqual(configured.FirstTokenUS, injected.FirstTokenUS) {
			t.Fatal("factory and configured joint policy execute differently", err)
		}
	})
	t.Run("ready batch balancing", func(t *testing.T) {
		c := restoreLabConfig(nil)
		c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 2, 64, 16
		c.Requests[2].ID = "a-hot"
		cold := c.Requests[2]
		cold.ID = "c-cold"
		cold.Input = append([]sim.TokenID(nil), cold.Input...)
		for i := range cold.Input {
			cold.Input[i] += 500
		}
		c.Requests = append(c.Requests, cold)
		c.DecisionPolicy = &DecisionPolicyConfig{ControlSteps: true, TraceEvents: true,
			BalancedBatch: &BalancedBatchConfig{MaxLoadComputeRatio: 8, TokenBudget: 16, MaxRequests: 2}}
		baseline, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		injected, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy {
			p, e := sim.NewBalancedBatchPolicyWithOptions(8, 16, 2, 0, sim.BalancedBatchOptions{ReadyComputeBudget: true})
			if e != nil {
				t.Fatal(e)
			}
			return p
		}})
		if err != nil {
			t.Fatal(err)
		}
		firstCold := func(r *Result) int64 {
			for _, d := range r.PolicyDecisions {
				for _, g := range d.Feedback.Grants {
					if g.Request == "c-cold" {
						return g.Tokens
					}
				}
			}
			return 0
		}
		if firstCold(baseline) != 15 || firstCold(injected) != 16 || len(injected.FinishedUS) != 4 {
			t.Fatal("factory did not change actual ready work", firstCold(baseline), firstCold(injected))
		}
		c.DecisionPolicy.BalancedBatch.ReadyComputeBudget = true
		configured, err := Run(c)
		if err != nil || !reflect.DeepEqual(configured.Events, injected.Events) || !reflect.DeepEqual(configured.FirstTokenUS, injected.FirstTokenUS) {
			t.Fatal("factory and configured balancing execute differently", err)
		}
		for _, r := range []*Result{baseline, injected, configured} {
			for _, p := range r.Pools {
				if p["reserved"] != 0 || p["read_pins"] != 0 {
					t.Fatal("joint policy leaked resources", p)
				}
			}
		}
	})
}
