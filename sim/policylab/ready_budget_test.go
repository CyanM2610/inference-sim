package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestReadyBudgetChangesActualColdChunkWhileKeepingRealLoads(t *testing.T) {
	for _, ready := range []bool{false, true} {
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
			BalancedBatch: &BalancedBatchConfig{MaxLoadComputeRatio: 8, TokenBudget: 16, MaxRequests: 2, ReadyComputeBudget: ready}}
		r, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		var first *sim.DecisionRecord
		for i := range r.PolicyDecisions {
			d := &r.PolicyDecisions[i]
			if d.View.NowUS >= 200000 && first == nil {
				first = d
			}
			var n int64
			for _, g := range d.Feedback.Grants {
				n += g.Tokens
			}
			if n > 16 {
				t.Fatal("actual grants exceeded policy budget", d)
			}
		}
		want := int64(15)
		if ready {
			want = 16
		}
		if first == nil || !reflect.DeepEqual(first.Plan.Admission.Requests, []string{"a-hot", "c-cold"}) ||
			!reflect.DeepEqual(first.Feedback.Grants, []sim.DecisionGrant{{Request: "c-cold", Tokens: want}}) {
			t.Fatalf("ready=%v: wrong actual first grant: %+v", ready, first)
		}
		var loads, blocks int64
		for _, e := range r.Events {
			if e.Name == "transfer_start" && e.Destination == "hbm" && e.Request == "a-hot" {
				loads++
				blocks += e.Bytes / r.BlockBytes
			}
		}
		if loads != 1 || blocks != 2 || len(r.FinishedUS) != 4 {
			t.Fatal("ready budgeting bypassed restoration or completion", loads, blocks, r.FinishedUS)
		}
	}
}
