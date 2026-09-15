package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type holdAdmissionOnce struct{ held bool }

func (p *holdAdmissionOnce) Decide(v sim.DecisionView) sim.DecisionPlan {
	plan := sim.DecisionPlan{Version: v.Version}
	for _, r := range v.Waiting {
		if r.ID == "reader" && !p.held {
			p.held = true
			at := v.NowUS + 1000
			plan.Admission = &sim.DecisionAdmission{Requests: []string{}}
			plan.RevisitAtUS = &at
		}
	}
	return plan
}
func (*holdAdmissionOnce) Observe(sim.DecisionFeedback) {}

func TestAdmissionSubsetGatesRealRestoreAndResetsNextStep(t *testing.T) {
	c := restoreLabConfig(nil)
	r, err := RunWithDecisionPolicy(c, func(string) sim.DecisionPolicy { return &holdAdmissionOnce{} })
	if err != nil {
		t.Fatal(err)
	}
	var rejected, restored bool
	for _, record := range r.PolicyDecisions {
		if record.Plan.Admission != nil {
			if len(record.Feedback.Grants) != 0 {
				t.Fatal("held request computed", record)
			}
			for _, u := range record.Feedback.Unselected {
				if u.Request == "reader" && u.Reason == "policy_not_admitted" {
					rejected = true
				}
			}
		}
	}
	for _, e := range r.Events {
		if e.Request == "reader" && e.Name == "transfer_start" && e.Destination == "hbm" {
			if e.Time < 201000 {
				t.Fatal("excluded request began a restore", e)
			}
			restored = true
		}
	}
	if !rejected || !restored || len(r.FinishedUS) != 3 {
		t.Fatal("admission did not block/reset with timer", rejected, restored, r.FinishedUS)
	}
}

func TestBalancedBatchConfigChangesAdmissionAndActualWork(t *testing.T) {
	fixture := func(ratio float64) Config {
		c := restoreLabConfig(nil)
		c.Instances[0].HBMBlocks = 6
		c.Pools[0].CapacityBlocks = 32
		c.MaxSequences = 2
		c.PrefillChunk = 16
		c.DecisionPolicy = &DecisionPolicyConfig{BalancedBatch: &BalancedBatchConfig{MaxLoadComputeRatio: ratio, TokenBudget: 16, MaxRequests: 2}, ControlSteps: true}
		tokens := func(base, n int) []sim.TokenID {
			a := make([]sim.TokenID, n)
			for i := range a {
				a[i] = sim.TokenID(base + i)
			}
			return a
		}
		c.Requests = []RequestConfig{
			{ID: "warm-a", At: 0, Input: tokens(1, 32), Output: []sim.TokenID{9}},
			{ID: "warm-b", At: 100000, Input: tokens(100, 32), Output: []sim.TokenID{9}},
			{ID: "evict", At: 200000, Input: tokens(1000, 96), Output: []sim.TokenID{9}},
			{ID: "a-hot", At: 300000, Input: tokens(1, 32), Output: []sim.TokenID{9}},
			{ID: "b-hot", At: 300000, Input: tokens(100, 32), Output: []sim.TokenID{9}},
			{ID: "c-cold", At: 300000, Input: tokens(2000, 32), Output: []sim.TokenID{9}},
		}
		return c
	}
	for _, ratio := range []float64{8, 1000} {
		c := fixture(ratio)
		r, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		want := []string{"a-hot", "c-cold"}
		if ratio == 1000 {
			want = []string{"a-hot", "b-hot"}
		}
		var first *sim.DecisionRecord
		var cappedRunning bool
		for i := range r.PolicyDecisions {
			d := &r.PolicyDecisions[i]
			if d.View.NowUS >= 300000 && first == nil {
				first = d
			}
			caps := map[string]int64{}
			for _, cap := range d.Plan.TokenCaps {
				caps[cap.Request] = cap.Tokens
			}
			var granted int64
			allowed := map[string]bool{}
			for _, id := range d.Plan.Admission.Requests {
				allowed[id] = true
			}
			waiting := map[string]bool{}
			for _, q := range d.View.Waiting {
				waiting[q.ID] = true
			}
			for _, g := range d.Feedback.Grants {
				granted += g.Tokens
				if g.Tokens > caps[g.Request] || waiting[g.Request] && !allowed[g.Request] {
					t.Fatal("runtime escaped admission/quota", *d)
				}
				for _, q := range d.View.Running {
					if q.ID == g.Request && q.ComputedTokens > 0 && q.ComputedTokens < q.InputTokens {
						cappedRunning = true
					}
				}
			}
			if granted > 16 {
				t.Fatal("aggregate quota exceeded", granted)
			}
		}
		if first == nil || !reflect.DeepEqual(first.Plan.Admission.Requests, want) {
			t.Fatalf("ratio %v selected wrong compute partner: %+v", ratio, first)
		}
		var coldFirst int64
		var hotBlocks int64
		for _, b := range r.BatchShapes {
			for _, w := range b.Scheduled {
				if w.Request == "c-cold" && coldFirst == 0 {
					coldFirst = w.Query
				}
			}
		}
		for _, e := range r.Events {
			if e.Name == "transfer_start" && e.Destination == "hbm" && (e.Request == "a-hot" || e.Request == "b-hot") {
				hotBlocks += e.Bytes / r.BlockBytes
			}
		}
		if ratio == 8 && coldFirst != 15 {
			t.Fatal("selected quota did not reach allocation", coldFirst)
		}
		if hotBlocks != 4 || len(r.FinishedUS) != len(c.Requests) || !cappedRunning {
			t.Fatal("missing physical loads/completion/running quota", hotBlocks, r.FinishedUS, cappedRunning)
		}
		for _, p := range r.Pools {
			if p["reserved"] != 0 || p["read_pins"] != 0 {
				t.Fatal("CPU leases leaked")
			}
		}
		for _, p := range r.HBM {
			if p["active_or_pinned"] != 0 {
				t.Fatal("HBM leases leaked")
			}
		}
	}
}
