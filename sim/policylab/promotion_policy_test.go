package policylab

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func promotionPolicyConfig() Config {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	c.PromotionControl = true
	c.DecisionPolicy = &DecisionPolicyConfig{TraceEvents: true, ExtraCost: &sim.LinearDecisionCost{FixedUS: 17, Provenance: "synthetic decision delay"}}
	c.Instances[0].HBMBlocks = 12
	c.Pools[0].CapacityBlocks = 64
	c.MaxSequences = 4
	c.TraceBatchShapes = true
	request := func(id string, at int64, length, group int) RequestConfig {
		input := make([]sim.TokenID, length)
		for i := range input {
			input[i] = sim.TokenID(group*1000 + i)
		}
		return RequestConfig{ID: id, At: at, Input: input, Output: []sim.TokenID{9}, MaxOutputTokens: 1}
	}
	c.Requests = []RequestConfig{request("seed", 0, 65, 1), request("evict", 10000, 177, 2),
		request("reader-a", 20000, 65, 1), request("reader-b", 20000, 65, 1), request("independent", 20001, 17, 3)}
	return c
}

func TestPromotionPolicyStartsLoadWithoutComputeAdmission(t *testing.T) {
	for _, retention := range []string{"cache", "request"} {
		t.Run(retention, func(t *testing.T) {
			c := promotionPolicyConfig()
			c.PromotionRetention = retention
			c.RestoreControl = retention == "request"
			checkPromotionPolicy(t, c)
		})
	}
}

func checkPromotionPolicy(t *testing.T, c Config) {
	t.Helper()
	r, err := RunWithDecisionPolicy(c, func(string) sim.DecisionPolicy {
		p, _ := sim.NewPrefetchQueuePolicy("fcfs")
		return p
	})
	if err != nil {
		t.Fatal(err)
	}
	var selected *sim.DecisionRecord
	var started, pending int64
	for i, d := range r.PolicyDecisions {
		for _, outcome := range d.Feedback.Promotions {
			started += outcome.StartedBlocks
			pending += outcome.PendingBlocks
			selected = &r.PolicyDecisions[i]
			for _, grant := range d.Feedback.Grants {
				if grant.Request == outcome.Request {
					t.Fatal("example admitted a request in its promotion decision")
				}
			}
		}
	}
	if selected == nil || started != 4 || pending != 4 || r.PromotionCostCoverage == "" {
		t.Fatalf("missing real/shared promotion evidence: started=%d pending=%d", started, pending)
	}
	var staged, adopted kv.PeerRecord
	for _, e := range r.Events {
		if e.Reason == "promotion" {
			if e.Name == "hbm_reserve" && e.Time < selected.Service.EndUS {
				t.Fatal("reservation preceded decision service completion")
			}
			if e.Name == "transfer_staged" {
				if staged.Transaction != 0 {
					t.Fatal("same prefix produced more than one physical copy")
				}
				staged = e
			}
			if e.Name == "transfer_adopted" {
				adopted = e
			}
		}
	}
	if staged.Transaction == 0 || staged.Transaction != adopted.Transaction || len(staged.Hashes) != 4 || staged.Time >= adopted.Time {
		t.Fatalf("staging/adoption evidence missing: %+v %+v", staged, adopted)
	}
	independentProgress := false
	for _, b := range r.BatchShapes {
		for _, work := range b.Scheduled {
			if work.Request == "reader-a" || work.Request == "reader-b" {
				if b.StartUS < adopted.Time || work.Prefix != 64 || work.Query != 1 {
					t.Fatal("consumer used unadopted KV or failed to reuse promoted prefix", b)
				}
			}
			if work.Request == "independent" && b.StartUS >= staged.Time && b.StartUS < adopted.Time {
				independentProgress = true
			}
		}
	}
	if !independentProgress || len(r.FirstTokenUS) != 5 || len(r.FinishedUS) != 5 {
		t.Fatal("independent request did not progress while promotion was pending")
	}
	observed := false
	for _, record := range r.PolicyEvents {
		e := record.Event
		if e.Transfer != nil && e.Transfer.ID == adopted.Transaction && e.Kind == "transfer_adopted" {
			observed = e.OccurredUS == adopted.Time && e.DeliveredUS >= e.OccurredUS
		}
	}
	if !observed || r.HBM["instance_0"]["active_or_pinned"] != 0 || r.Pools["dram"]["read_pins"] != 0 {
		t.Fatal("missing adopted notification or leaked resources")
	}
}

func TestPromotionControlRequiresExplicitExecutionScope(t *testing.T) {
	for _, mode := range []string{"decision", "engine", "grouping", "offload", "retention", "retention_without_promotion", "request_without_restore"} {
		c := promotionPolicyConfig()
		switch mode {
		case "decision":
			c.DecisionPolicy = nil
		case "engine":
			c.EnginePhases = nil
		case "grouping":
			c.Mechanisms.GroupTransfers = false
		case "offload":
			c.Pools = nil
		case "retention":
			c.PromotionRetention = "unknown"
		case "retention_without_promotion":
			c.PromotionRetention, c.PromotionControl = "cache", false
		case "request_without_restore":
			c.PromotionRetention, c.RestoreControl = "request", false
		}
		if c.Validate() == nil {
			t.Fatal("unsupported promotion scope accepted", mode)
		}
	}
}
