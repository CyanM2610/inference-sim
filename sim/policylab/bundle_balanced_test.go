package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestBundleSelectionRunsThroughSharedLoadAndCapacityChecks(t *testing.T) {
	for _, capacity := range []int64{4, 6} {
		for _, bundle := range []bool{false, true} {
			c := restoreLabConfig(nil)
			c.Instances[0].HBMBlocks = capacity
			c.Requests[1].Input = make([]sim.TokenID, capacity*16)
			for i := range c.Requests[1].Input {
				c.Requests[1].Input[i] = sim.TokenID(100 + i)
			}
			c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 2, 16, 16
			c.DecisionPolicy = &DecisionPolicyConfig{PrefixSharing: true, ControlSteps: true, TraceEvents: true,
				BalancedBatch: &BalancedBatchConfig{MaxLoadComputeRatio: 8, TokenBudget: 16, MaxRequests: 2, BundleHits: bundle}}
			c.Requests[2].ID = "a-hot"
			c.Requests[0].Input = append(c.Requests[0].Input, 900)
			c.Requests[2].Input = append(c.Requests[2].Input, 900)
			b := c.Requests[2]
			b.ID = "b-hot"
			cold := c.Requests[2]
			cold.ID = "c-cold"
			cold.Input = append([]sim.TokenID(nil), cold.Input...)
			for i := range cold.Input {
				cold.Input[i] += 500
			}
			c.Requests = append(c.Requests, b, cold)
			r, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			want := []string{"a-hot", "c-cold"}
			if bundle {
				want = []string{"a-hot", "b-hot"}
			}
			var first *sim.DecisionRecord
			for i := range r.PolicyDecisions {
				d := &r.PolicyDecisions[i]
				if d.View.NowUS >= 200000 && first == nil {
					first = d
				}
				var total int64
				for _, g := range d.Feedback.Grants {
					total += g.Tokens
				}
				if total > 16 {
					t.Fatal("bundle escaped runtime token budget", d)
				}
			}
			if first == nil || !reflect.DeepEqual(first.Plan.Admission.Requests, want) {
				t.Fatalf("capacity=%d bundle=%v wrong initial admission: %+v", capacity, bundle, first)
			}
			var loads int
			var loaded int64
			for _, e := range r.Events {
				if e.Time >= 200000 && e.Name == "transfer_start" && e.Destination == "hbm" && (e.Request == "a-hot" || e.Request == "b-hot") {
					loads++
					loaded += e.Bytes / r.BlockBytes
				}
			}
			if bundle && (loads != 1 || loaded != 2) {
				t.Fatalf("bundle did not share one physical load: jobs=%d blocks=%d", loads, loaded)
			}
			if len(r.FinishedUS) != 5 {
				t.Fatal("bundle/capacity fixture did not drain", r.FinishedUS)
			}
			for _, pool := range r.Pools {
				if pool["reserved"] != 0 || pool["read_pins"] != 0 {
					t.Fatal("shared loads leaked CPU leases", pool)
				}
			}
			for _, hbm := range r.HBM {
				if hbm["active_or_pinned"] != 0 {
					t.Fatal("shared loads leaked HBM leases", hbm)
				}
			}
		}
	}
}
