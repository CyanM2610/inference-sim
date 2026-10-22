package policylab

import (
	"math"
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

func TestConversationForkJoinFollowsAllActualCompletions(t *testing.T) {
	c := conversationFixture()
	c.Requests[1].AfterRequest = ""
	c.Requests[1].AfterRequests = []string{"a0"}
	c.Requests[2].AfterRequest = ""
	c.Requests[2].AfterRequests = []string{"a1", "b0"}
	c.Requests[2].ThinkTimeUS = 71
	c.Requests[3].At = 0
	c.Requests[3].AfterRequests = []string{"a0"}
	c.Requests[3].ThinkTimeUS = 37
	c.Requests[3].Output = make([]sim.TokenID, 15)
	for i := range c.Requests[3].Output {
		c.Requests[3].Output[i] = sim.TokenID(100 + i)
	}
	c.Requests[3].MaxOutputTokens = 16
	for _, delay := range []int64{0, 200} {
		policy := *c.DecisionPolicy
		policy.AdmissionDelayUS = delay
		c.DecisionPolicy = &policy
		r, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		if len(r.FinishedUS) != len(c.Requests) || r.ArrivalUS["a0"] != 10 {
			t.Fatal("lost fork/join request", r.ArrivalUS)
		}
		if r.ArrivalUS["a1"] != r.FinishedUS["a0"]+37 || r.ArrivalUS["b0"] != r.ArrivalUS["a1"] {
			t.Fatal("fork did not release siblings together", r.ArrivalUS)
		}
		last := max(r.FinishedUS["a1"], r.FinishedUS["b0"])
		if r.ArrivalUS["a2"] != last+71 {
			t.Fatal("join released before every parent completed", r.ArrivalUS, r.FinishedUS)
		}
		for _, record := range r.PolicyDecisions {
			visible := append(append([]sim.DecisionRequest{}, record.View.Waiting...), record.View.Running...)
			for _, req := range visible {
				if r.ArrivalUS[req.ID] > record.View.NowUS {
					t.Fatal("unreleased DAG request leaked to policy", req.ID)
				}
			}
		}
	}
	if c.Requests[2].At != 0 {
		t.Fatal("mutated frozen arrival dependencies")
	}
}

func TestConversationForkJoinValidation(t *testing.T) {
	for _, change := range []func(*Config){
		func(c *Config) { c.Requests[1].AfterRequests = []string{"a0"} }, // ambiguous singular/plural
		func(c *Config) { c.Requests[1].AfterRequest = ""; c.Requests[1].AfterRequests = []string{"a0", "a0"} },
		func(c *Config) { c.Requests[1].AfterRequest = ""; c.Requests[1].AfterRequests = []string{"missing"} },
		func(c *Config) { c.Requests[1].AfterRequest = ""; c.Requests[1].AfterRequests = []string{"a1"} },
		func(c *Config) { c.Requests[0].At = 0; c.Requests[0].AfterRequests = []string{"a2", "b0"} },
		func(c *Config) {
			c.Requests[1].AfterRequest = ""
			c.Requests[1].AfterRequests = []string{"a0"}
			c.Requests[1].At = 1
		},
	} {
		c := conversationFixture()
		change(&c)
		if err := c.Validate(); err == nil {
			t.Fatal("invalid completion DAG accepted")
		}
	}
}

func TestConversationJoinCompletionAccounting(t *testing.T) {
	for _, reverse := range []bool{false, true} {
		c := conversationFixture()
		c.Requests[1].AfterRequest = ""
		c.Requests[1].AfterRequests = []string{"a0", "b0"}
		config := map[string]RequestConfig{}
		var requests []*sim.Request
		for _, r := range c.Requests {
			config[r.ID] = r
			requests = append(requests, &sim.Request{ID: r.ID, State: sim.StateCompleted})
		}
		s := newConversationArrivals(c, requests, config, map[string]int64{})
		first, second := requests[0], requests[3]
		if reverse {
			first, second = second, first
		}
		if got := s.complete(first, 100); len(got) != 0 {
			t.Fatal("partial join released", got)
		}
		got := s.complete(second, 200)
		if s.err != nil || len(got) != 1 || got[0].ID != "a1" || got[0].ArrivalTime != 237 || s.deadlines["a1"] != 10237 {
			t.Fatal("bad release or SLO deadline", got, s.err, s.deadlines)
		}
		if got := s.complete(first, 250); len(got) != 0 || s.err == nil {
			t.Fatal("duplicate completion accepted")
		}
	}
	c := conversationFixture()
	s := newConversationArrivals(c, []*sim.Request{{ID: "a1"}}, map[string]RequestConfig{"a1": c.Requests[1]}, map[string]int64{})
	if got := s.complete(&sim.Request{ID: "a0", State: sim.StateCompleted}, math.MaxInt64-1); len(got) != 0 || s.err == nil {
		t.Fatal("overflow accepted")
	}
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
