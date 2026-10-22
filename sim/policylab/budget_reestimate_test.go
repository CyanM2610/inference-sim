package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func reestimateExecutionConfig(enabled bool) Config {
	c := pressureBudgetConfig()
	c.DecisionPolicy.BudgetRestore = &BudgetRestoreConfig{Mode: "budget_queues", Guardband: 1,
		PrimaryWeight: 4, BestEffortWeight: 1, ReestimateOnDecodeDrop: enabled}
	c.Instances[0].HBMBlocks, c.Pools[0].CapacityBlocks = 100, 128
	c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 2, 7, 7
	c.Requests = nil
	for i, id := range []string{"decoder", "primary", "best"} {
		length, arrival, slo := 512, int64(1), int64(1000000000)
		outputs := []sim.TokenID{9, 10}
		if id == "decoder" {
			length, arrival = 1, 0
			outputs = append(outputs, 11)
		}
		if id == "best" {
			arrival, slo = 26, 7000
		}
		input := make([]sim.TokenID, length)
		for j := range input {
			input[j] = sim.TokenID(10000*(i+1) + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: id, At: arrival, Input: input,
			Output: outputs, MaxOutputTokens: len(outputs), TTFTSLOUS: slo})
	}
	return c
}

func TestBudgetReestimateChangesActualRuntimeService(t *testing.T) {
	c := reestimateExecutionConfig(true)
	configured, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	injected, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy {
		p, e := sim.NewBudgetQueuePolicyWithOptions(1, 0, 4, 1, sim.BudgetQueueOptions{ReestimateOnDecodeDrop: true})
		if e != nil {
			t.Fatal(e)
		}
		return &boundedBudgetQueue{BudgetQueuePolicy: p, t: t}
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(configured.Events, injected.Events) || !reflect.DeepEqual(configured.FirstTokenUS, injected.FirstTokenUS) {
		t.Fatal("configuration and public factory disagree")
	}
	baseline, err := Run(reestimateExecutionConfig(false))
	if err != nil {
		t.Fatal(err)
	}
	replay, _ := sim.NewBudgetQueuePolicyWithOptions(1, 0, 4, 1, sim.BudgetQueueOptions{ReestimateOnDecodeDrop: true})
	var original *sim.RequestBudget
	var promotedGrants int64
	for _, d := range configured.PolicyDecisions {
		if !reflect.DeepEqual(replay.Decide(d.View), d.Plan) {
			t.Fatalf("actual plan replay differs at version %d", d.View.Version)
		}
		for _, b := range replay.RequestBudgets(d.View.NowUS) {
			if b.Request != "best" || !b.Known {
				continue
			}
			if original == nil {
				first := b
				original = &first
				t.Logf("initial best budget: %+v", b)
			}
			if b.InitialUS != original.InitialUS || b.InitializedUS != original.InitializedUS || b.ArrivalUS != 26 {
				t.Fatal("reestimate rewrote initial history", b)
			}
			if b.ReestimateCount > 0 && b.RemainingUS > 0 {
				for _, g := range d.Feedback.Grants {
					if g.Request == b.Request {
						promotedGrants += g.Tokens
					}
				}
			}
		}
		replay.Observe(d.Feedback)
	}
	if original == nil || original.RemainingUS > 0 || promotedGrants == 0 {
		t.Fatalf("fixture did not execute a best-effort promotion: initial=%+v promoted grants=%d TTFT=%v", original, promotedGrants, configured.FirstTokenUS)
	}
	// Promotion changes actual sharing; it need not improve mean TTFT or the
	// promoted request's final TTFT when its remaining budget expires again.
	if reflect.DeepEqual(configured.BatchShapes, baseline.BatchShapes) || reflect.DeepEqual(configured.FirstTokenUS, baseline.FirstTokenUS) {
		t.Fatalf("reclassification did not change actual service: enabled=%v disabled=%v", configured.FirstTokenUS, baseline.FirstTokenUS)
	}
	for _, result := range []*Result{configured, injected, baseline} {
		if len(result.FinishedUS) != 3 || len(result.FirstTokenUS) != 3 {
			t.Fatal("incomplete workload")
		}
		for _, pool := range result.Pools {
			if pool["reserved"] != 0 || pool["read_pins"] != 0 {
				t.Fatal("pool lease leaked", pool)
			}
		}
		for _, hbm := range result.HBM {
			if hbm["active_or_pinned"] != 0 || hbm["decode_reserved_blocks"] != 0 || hbm["decode_capacity_requests"] != 0 {
				t.Fatal("HBM lease leaked", hbm)
			}
		}
	}
	if configured.DecisionAdditionalCostCoverage != "budget_queue_reestimate_not_independently_calibrated" {
		t.Fatal("missing new work cost scope")
	}
}

func TestBudgetReestimateWithoutEligibleDropPreservesExecution(t *testing.T) {
	results := make([]*Result, 2)
	for i, enabled := range []bool{false, true} {
		c := reestimateExecutionConfig(enabled)
		c.Requests[0].Output, c.Requests[0].MaxOutputTokens = []sim.TokenID{9}, 1
		var err error
		results[i], err = Run(c)
		if err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(results[0].Events, results[1].Events) || !reflect.DeepEqual(results[0].FirstTokenUS, results[1].FirstTokenUS) {
		t.Fatal("unobserved decoder must not change execution")
	}
}
