package policylab

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestInitialBudgetUsesCommonRestoreRuntime(t *testing.T) {
	for _, tc := range []struct{ slo, blocks int64 }{{120, 0}, {180, 1}, {400, 2}} {
		t.Run(fmt.Sprint(tc.slo), func(t *testing.T) {
			c := budgetLabConfig()
			c.DecisionPolicy.BudgetRestore.Mode = "initial_budget"
			c.Requests[2].TTFTSLOUS = tc.slo
			r, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			if r.DecisionAdditionalCostCoverage != "initial_budget_history_not_independently_calibrated" {
				t.Fatal("new history costs silently inherited old calibration")
			}
			var bytes, compute int64
			for _, e := range r.Events {
				if e.Name == "transfer_start" && e.Request == "reader" && e.Destination == "hbm" {
					bytes += e.Bytes
				}
			}
			for _, b := range r.BatchShapes {
				for _, q := range b.Scheduled {
					if q.Request == "reader" {
						compute += q.Query
					}
				}
			}
			if bytes != tc.blocks*r.BlockBytes || compute != max(int64(1), 32-16*tc.blocks) || len(r.FinishedUS) != 3 {
				t.Fatal("persistent budget did not drive actual restore/recompute", bytes, compute)
			}
			// Replay each actual observation in order; a new object per snapshot
			// would erase the very history this contract is intended to verify.
			p, _ := sim.NewInitialBudgetRestorePolicy(1, 0)
			for _, record := range r.PolicyDecisions {
				if got := p.Decide(record.View); !reflect.DeepEqual(got, record.Plan) {
					t.Fatal("runtime and sequential replay differ", got, record.Plan)
				}
				p.Observe(record.Feedback)
			}
			for _, pool := range r.Pools {
				if pool["reserved"] != 0 || pool["read_pins"] != 0 {
					t.Fatal("pool reservation leaked")
				}
			}
			for _, hbm := range r.HBM {
				if hbm["active_or_pinned"] != 0 {
					t.Fatal("HBM reservation leaked")
				}
			}
		})
	}
}

func TestInitialBudgetPressureKeepsActualRequestHistory(t *testing.T) {
	c := budgetLabConfig()
	c.DecisionPolicy.BudgetRestore.Mode = "initial_budget"
	c.Instances[0].HBMBlocks, c.MaxSequences, c.PrefillChunk = 10, 4, 16
	c.Requests = nil
	for i := 0; i < 24; i++ {
		tokens := make([]sim.TokenID, 32+16*(i%3))
		for j := range tokens {
			tokens[j] = sim.TokenID(1000*(i%4) + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i * 20), Input: tokens, Output: []sim.TokenID{9, 10, 11, 12}, TTFTSLOUS: 2000})
	}
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.FinishedUS) != 24 || len(r.FirstTokenUS) != 24 {
		t.Fatal("persistent budget stranded pressure workload")
	}
	p, _ := sim.NewInitialBudgetRestorePolicy(1, 0)
	initial := map[string]sim.RequestBudget{}
	runningKnown := 0
	for _, d := range r.PolicyDecisions {
		if !reflect.DeepEqual(p.Decide(d.View), d.Plan) {
			t.Fatal("pressure replay disagreed")
		}
		rows := p.RequestBudgets(d.View.NowUS)
		for _, b := range rows {
			if !b.Known {
				continue
			}
			if old, ok := initial[b.Request]; ok && (old.InitialUS != b.InitialUS || old.InitializedUS != b.InitializedUS) {
				t.Fatal("actual runtime observation reset initial budget")
			}
			initial[b.Request] = b
			for _, active := range d.View.Running {
				if active.ID == b.Request && active.ComputedTokens < active.InputTokens {
					runningKnown++
				}
			}
		}
	}
	if len(initial) != 24 || runningKnown == 0 {
		t.Fatal("test did not observe initialized running prefills", len(initial), runningKnown)
	}
}

func TestInitialBudgetPrefixGrowthChangesActualAdmission(t *testing.T) {
	c := budgetLabConfig()
	c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 1, 16, 16
	c.Instances[0].HBMBlocks = 20
	tokens := func(n, base int) []sim.TokenID {
		r := make([]sim.TokenID, n)
		for i := range r {
			r[i] = sim.TokenID(base + i)
		}
		return r
	}
	c.Requests = []RequestConfig{
		{ID: "producer", At: 0, Input: tokens(128, 1), Output: []sim.TokenID{9}},
		{ID: "long", At: 10, Input: tokens(128, 1), Output: []sim.TokenID{9}, TTFTSLOUS: 5000},
		{ID: "short", At: 10, Input: tokens(32, 1000), Output: []sim.TokenID{9}, TTFTSLOUS: 5000},
	}
	for _, mode := range []string{"restore_bound", "initial_budget"} {
		c.DecisionPolicy.BudgetRestore.Mode = mode
		r, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		var first string
		for _, b := range r.BatchShapes {
			for _, q := range b.Scheduled {
				if q.Request != "producer" && first == "" {
					first = q.Request
				}
			}
		}
		want := "short"
		if mode == "initial_budget" {
			want = "long"
		}
		if first != want || len(r.FirstTokenUS) != 3 {
			t.Fatal("prefix-growth counterexample did not change actual admission", mode, first, r.FirstTokenUS)
		}
	}
}
