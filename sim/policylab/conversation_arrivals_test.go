package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func conversationFixture() Config {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	c.DecisionPolicy = &DecisionPolicyConfig{QueueOrder: "fcfs", ControlSteps: true, TraceEvents: true}
	c.Requests = []RequestConfig{
		{ID: "a0", At: 10, Input: []sim.TokenID{1, 2, 3}, Output: []sim.TokenID{4, 5}, MaxOutputTokens: 8, TTFTSLOUS: 10000},
		{ID: "a1", AfterRequest: "a0", ThinkTimeUS: 37, Input: []sim.TokenID{1, 2, 3, 4, 5, 6}, Output: []sim.TokenID{7, 8}, MaxOutputTokens: 8, TTFTSLOUS: 10000},
		{ID: "a2", AfterRequest: "a1", Input: []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}, Output: []sim.TokenID{10}, MaxOutputTokens: 8, TTFTSLOUS: 10000},
		{ID: "b0", At: 17, Input: []sim.TokenID{21, 22}, Output: []sim.TokenID{23}, MaxOutputTokens: 8},
	}
	return c
}

func TestConversationArrivalsFollowActualCompletionAndStayHidden(t *testing.T) {
	c := conversationFixture()
	first, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	delayed := c
	policy := *c.DecisionPolicy
	policy.AdmissionDelayUS = 200
	delayed.DecisionPolicy = &policy
	second, err := Run(delayed)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(c.Requests, delayed.Requests) || c.Requests[1].At != 0 {
		t.Fatal("external inputs mutated")
	}
	for _, r := range []*Result{first, second} {
		if len(r.FinishedUS) != 4 || r.ArrivalUS["a0"] != 10 || r.ArrivalUS["b0"] != 17 {
			t.Fatal("lost turn/root arrival", r.ArrivalUS)
		}
		if r.ArrivalUS["a1"] != r.FinishedUS["a0"]+37 || r.ArrivalUS["a2"] != r.FinishedUS["a1"] {
			t.Fatal("follow-up anchored to predicted time", r.ArrivalUS, r.FinishedUS)
		}
		for id, at := range r.ArrivalUS {
			if r.FirstTokenUS[id] < at || r.FirstTokenUS[id] > r.FinishedUS[id] {
				t.Fatal("wrong TTFT origin", id)
			}
		}
		for _, record := range r.PolicyDecisions {
			visible := append(append([]sim.DecisionRequest{}, record.View.Waiting...), record.View.Running...)
			for _, req := range visible {
				if r.ArrivalUS[req.ID] > record.View.NowUS {
					t.Fatal("future turn leaked to policy", req.ID)
				}
			}
		}
	}
	if second.ArrivalUS["a1"] <= first.ArrivalUS["a1"] {
		t.Fatal("different policy speed did not change endogenous arrival")
	}
}

func TestConversationArrivalsRejectInvalidChains(t *testing.T) {
	for _, change := range []func(*Config){
		func(c *Config) { c.Requests[1].AfterRequest = "missing" },
		func(c *Config) { c.Requests[0].At = 0; c.Requests[0].AfterRequest = "a2" },
		func(c *Config) { c.Requests[2].AfterRequest = "a0" },
		func(c *Config) { c.Requests[1].At = 1 },
		func(c *Config) { c.Requests[1].ThinkTimeUS = -1 },
		func(c *Config) { c.Requests[0].ThinkTimeUS = 2 },
	} {
		c := conversationFixture()
		change(&c)
		if err := c.Validate(); err == nil {
			t.Fatal("invalid conversation accepted")
		}
	}
}
