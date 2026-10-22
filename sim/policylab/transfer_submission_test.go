package policylab

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

type reverseSubmissionPolicy struct{ calls int }

func (p *reverseSubmissionPolicy) Order(c kv.TransferSubmissionContext) []int64 {
	p.calls++
	ids := make([]int64, len(c.Candidates))
	for i, job := range c.Candidates {
		ids[len(ids)-1-i] = job.Transaction
	}
	return ids
}

func submissionLabConfig() Config {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	c.DecisionPolicy = &DecisionPolicyConfig{}
	for i := range c.Requests {
		c.Requests[i].At = int64(i/2) * 100000
	}
	return c
}

func TestTransferSubmissionFactoryComposesAndExecutes(t *testing.T) {
	c := submissionLabConfig()
	c.DirectionalTransferOrder = true
	reqBase, _ := sim.NewQueueDecisionPolicy("sjf", 0)
	req := &countedDecision{base: reqBase}
	peer := &selectedPoolPolicy{pool: "dram"}
	pool := &countedPoolPolicy{}
	transfer := &reverseSubmissionPolicy{}
	created := 0
	r, err := RunWithPolicies(c, PolicyFactories{
		Decision:     func(string) sim.DecisionPolicy { return req },
		Peer:         func(string) kv.PeerPolicy { return peer },
		PoolEviction: func(string) kv.PoolEvictionPolicy { return pool },
		TransferSubmission: func(id string) kv.TransferSubmissionPolicy {
			created++
			if id != "instance_0" {
				t.Fatal(id)
			}
			return transfer
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if created != 1 || transfer.calls == 0 || transfer.calls != len(r.TransferSubmissions) || req.calls == 0 || peer.calls == 0 || pool.victims == 0 {
		t.Fatalf("missing actual composition: created=%d calls=%d req=%d peer=%d pool=%d", created, transfer.calls, req.calls, peer.calls, pool.victims)
	}
	multi := 0
	for _, record := range r.TransferSubmissions {
		if len(record.Order) > 1 {
			multi++
		}
		for i, id := range record.Order {
			if id != record.View.Candidates[len(record.Order)-1-i].Transaction {
				t.Fatal("factory order not used")
			}
		}
		var actual []int64
		for _, e := range r.Events {
			if e.Name == "transfer_submit_begin" && e.Time >= record.ServiceEndUS && e.Time < record.SubmitEndUS {
				actual = append(actual, e.Transaction)
			}
		}
		if !reflect.DeepEqual(actual, record.Order) {
			t.Fatal("decision did not reach actual engine submit", actual, record.Order)
		}
	}
	if multi == 0 || len(r.InjectedPolicies) != 4 || r.TransferSubmissionCostCoverage == "" || len(r.FirstTokenUS) != len(c.Requests) {
		t.Fatal("missing multi-job phase/coverage/completion", multi)
	}
	if r.DirectionalTransferOrderCoverage == "" || r.Counts["transfer_submission_dependency"] == 0 {
		t.Fatal("directional dependency model not installed")
	}
	for _, state := range r.Pools {
		if state["reserved"] != 0 || state["read_pins"] != 0 {
			t.Fatal("leaked transfer leases")
		}
	}
	if r.HBM["instance_0"]["active_or_pinned"] != 0 {
		t.Fatal("leaked HBM")
	}
}

func TestTransferSubmissionConfiguredFIFOPreservesDefault(t *testing.T) {
	c := submissionLabConfig()
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.TransferSubmissionPolicy = "fifo"
	fifo, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Events, fifo.Events) || !reflect.DeepEqual(base.FirstTokenUS, fifo.FirstTokenUS) || !reflect.DeepEqual(base.FinishedUS, fifo.FinishedUS) || !reflect.DeepEqual(base.Pools, fifo.Pools) || !reflect.DeepEqual(base.HBM, fifo.HBM) {
		t.Fatal("FIFO changed default execution")
	}
	if len(base.TransferSubmissions) != 0 || base.TransferSubmissionCostCoverage != "" || len(fifo.TransferSubmissions) == 0 {
		t.Fatal("missing optional observation")
	}
	c.TransferSubmissionCost = &kv.TransferSubmissionCost{FixedUS: 100, PerJobUS: 10, Provenance: "synthetic fixture"}
	paid, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range paid.TransferSubmissions {
		if record.ServiceEndUS-record.View.NowUS != 100+10*int64(len(record.Order)) {
			t.Fatal("declared cost not applied", record)
		}
	}
	if reflect.DeepEqual(fifo.FinishedUS, paid.FinishedUS) {
		t.Fatal("positive cost did not reach request timeline")
	}
}

func TestTransferSubmissionRejectsUnsupportedSetupBeforeFactory(t *testing.T) {
	for _, change := range []func(*Config){
		func(c *Config) { c.DirectionalTransferOrder = true; c.EnginePhases = nil },
		func(c *Config) { c.EnginePhases = nil },
		func(c *Config) { c.TransferSubmissionPolicy = "bogus" },
		func(c *Config) { c.TransferSubmissionCost = &kv.TransferSubmissionCost{FixedUS: 1} },
	} {
		c := submissionLabConfig()
		change(&c)
		called := false
		_, err := RunWithPolicies(c, PolicyFactories{TransferSubmission: func(string) kv.TransferSubmissionPolicy { called = true; return &reverseSubmissionPolicy{} }})
		if err == nil || called {
			t.Fatal("invalid setup reached factory", err)
		}
	}
	for _, policy := range []kv.TransferSubmissionPolicy{nil, (*reverseSubmissionPolicy)(nil)} {
		_, err := RunWithPolicies(submissionLabConfig(), PolicyFactories{TransferSubmission: func(string) kv.TransferSubmissionPolicy { return policy }})
		if err == nil || !strings.Contains(err.Error(), "nil") {
			t.Fatal("accepted nil factory", err)
		}
	}
	c := submissionLabConfig()
	c.TransferSubmissionCost = &kv.TransferSubmissionCost{Provenance: "fixture"}
	if _, err := Run(c); err == nil {
		t.Fatal("orphan cost accepted")
	}
	for _, configured := range []bool{true, false} {
		for _, c := range []Config{queueCostFixture(0), decodeFillCostFixture()} {
			factories := PolicyFactories{}
			if configured {
				c.TransferSubmissionPolicy = "fifo"
			} else {
				factories.TransferSubmission = func(string) kv.TransferSubmissionPolicy { return &reverseSubmissionPolicy{} }
			}
			_, err := RunWithPolicies(c, factories)
			if err == nil || !strings.Contains(err.Error(), "calibrate") {
				t.Fatal("new submission policy inherited old calibration", err)
			}
		}
	}
}
