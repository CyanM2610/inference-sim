package policylab

import (
	"github.com/inference-sim/inference-sim/sim"
	"reflect"
	"testing"
)

func producerLabConfig() Config {
	c := restoreLabConfig(nil)
	c.Instances[0].HBMBlocks = 24
	c.Pools[0].CapacityBlocks = 48
	c.MaxBatchTokens = 64
	c.PrefillChunk = 16
	c.MaxSequences = 4
	c.DecisionPolicy = &DecisionPolicyConfig{BalancedBatch: &BalancedBatchConfig{MaxLoadComputeRatio: 100, TokenBudget: 64, MaxRequests: 4}, ProducerPrefixes: true, ControlSteps: true, TraceEvents: true}
	a := make([]sim.TokenID, 64)
	for i := range a {
		a[i] = sim.TokenID(i + 1)
	}
	b := append([]sim.TokenID(nil), a...)
	b[63] = 1000
	cold := make([]sim.TokenID, 64)
	for i := range cold {
		cold[i] = sim.TokenID(i + 2000)
	}
	c.Requests = []RequestConfig{{ID: "a-producer", At: 0, Input: a, Output: []sim.TokenID{9}},
		{ID: "b-consumer", At: 1, Input: b, Output: []sim.TokenID{9}},
		{ID: "c-independent", At: 1, Input: cold, Output: []sim.TokenID{9}}}
	return c
}

func TestProducerWaitDefersComputeUntilPublishedPrefix(t *testing.T) {
	c := producerLabConfig()
	baseline, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.ProducerWait = &ProducerWaitConfig{MinSharedTokens: 32, MaxWaitUS: 1000000}
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	work := func(r *Result, id string) int64 {
		var n int64
		for _, b := range r.BatchShapes {
			for _, w := range b.Scheduled {
				if w.Request == id {
					n += w.Query
				}
			}
		}
		return n
	}
	if work(r, "b-consumer") != 16 || work(baseline, "b-consumer") <= 16 {
		t.Fatal("waiting did not eliminate redundant prefill", work(r, "b-consumer"), work(baseline, "b-consumer"))
	}
	var held, independent bool
	for _, d := range r.PolicyDecisions {
		for _, u := range d.Feedback.Unselected {
			if u.Request == "b-consumer" && u.Reason == "policy_not_admitted" {
				held = true
				for _, g := range d.Feedback.Grants {
					if g.Request == "c-independent" {
						independent = true
					}
				}
			}
		}
		for _, g := range d.Feedback.Grants {
			if g.Request == "b-consumer" {
				for _, state := range d.View.KV.Requests {
					if state.ID == g.Request && state.LocalPrefixBlocks < 3 {
						t.Fatal("consumer ran before actual ready prefix", d)
					}
				}
			}
		}
	}
	if !held || !independent || len(r.FinishedUS) != 3 {
		t.Fatal("hold/independent progress/completion missing", held, independent, r.FinishedUS)
	}
	for _, h := range r.HBM {
		if h["active_or_pinned"] != 0 {
			t.Fatal("HBM lease leaked")
		}
	}
	for _, p := range r.Pools {
		if p["reserved"] != 0 || p["read_pins"] != 0 {
			t.Fatal("pool lease leaked")
		}
	}
	// Exposing the extra view alone must not change execution or TTFT.
	c.DecisionPolicy.ProducerWait = nil
	c.DecisionPolicy.ProducerPrefixes = false
	off, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(off.Events, baseline.Events) || !reflect.DeepEqual(off.BatchShapes, baseline.BatchShapes) || !reflect.DeepEqual(off.FirstTokenUS, baseline.FirstTokenUS) {
		t.Fatal("read-only producer matching changed execution")
	}
}

func TestProducerAgeLimitFallsBackToImmediatePath(t *testing.T) {
	c := producerLabConfig()
	base, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.ProducerWait = &ProducerWaitConfig{MinSharedTokens: 32, MaxWaitUS: 1}
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.BatchShapes, r.BatchShapes) || !reflect.DeepEqual(base.FirstTokenUS, r.FirstTokenUS) {
		t.Fatal("expired waiting did not use normal execution")
	}
}
