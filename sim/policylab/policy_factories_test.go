package policylab

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

type countedDecision struct {
	base  sim.DecisionPolicy
	calls int
}

func (p *countedDecision) Decide(v sim.DecisionView) sim.DecisionPlan {
	p.calls++
	return p.base.Decide(v)
}
func (p *countedDecision) Observe(f sim.DecisionFeedback) { p.base.Observe(f) }

type selectedPoolPolicy struct {
	pool  string
	calls int
}

func (p *selectedPoolPolicy) Choose(c kv.PeerReclaimContext) kv.PeerDecision {
	p.calls++
	d := kv.PeerDecision{BlockID: c.Candidates[0].ID}
	for _, target := range c.Targets {
		if target.Pool == p.pool && target.Available {
			d.Pool = p.pool
			break
		}
	}
	return d
}

type countedPoolPolicy struct {
	kv.PrefixLRU
	events, victims int
}

func (p *countedPoolPolicy) Observe(e kv.PoolPolicyEvent) {
	p.events++
	p.PrefixLRU.Observe(e)
}
func (p *countedPoolPolicy) Victim(c []kv.PoolEvictionCandidate) string {
	p.victims++
	return p.PrefixLRU.Victim(c)
}

func TestPolicyFactoriesComposeRequestAndKVDecisions(t *testing.T) {
	// Requests arrive while a blocker is running. Both request orders must
	// exercise the injected save target and eviction under real pool pressure.
	c := labConfig(1, "both")
	c.DecisionPolicy = &DecisionPolicyConfig{}
	c.MaxSequences = 1
	for i := range c.Pools {
		c.Pools[i].CapacityBlocks = 2
	}
	request := func(id string, at int64, length, group int) RequestConfig {
		input := make([]sim.TokenID, length)
		for i := range input {
			input[i] = sim.TokenID(group*1000 + i)
		}
		return RequestConfig{ID: id, At: at, Input: input, Output: []sim.TokenID{9}}
	}
	c.Requests = []RequestConfig{request("blocker", 0, 65, 1), request("long", 1, 97, 2), request("short", 2, 33, 3),
		request("late-long", 1000000, 97, 4), request("late-short", 1000001, 33, 5), request("last", 2000000, 65, 6)}
	for _, order := range []string{"fcfs", "sjf"} {
		for _, pool := range []string{"dram", "cxl"} {
			t.Run(order+"/"+pool, func(t *testing.T) {
				base, err := sim.NewQueueDecisionPolicy(order, 0)
				if err != nil {
					t.Fatal(err)
				}
				req := &countedDecision{base: base}
				peer := &selectedPoolPolicy{pool: pool}
				pools := map[string]*countedPoolPolicy{}
				r, err := RunWithPolicies(c, PolicyFactories{
					Decision: func(string) sim.DecisionPolicy { return req },
					Peer:     func(string) kv.PeerPolicy { return peer },
					PoolEviction: func(id string) kv.PoolEvictionPolicy {
						p := &countedPoolPolicy{}
						pools[id] = p
						return p
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				var scheduled []string
				seen := map[string]bool{}
				for _, event := range r.Events {
					// Capacity pressure can preempt and re-admit a request.
					// Compare first admissions, keeping all raw events in Result.
					if event.Name == "sim.ScheduledEvent" && !seen[event.Request] {
						scheduled = append(scheduled, event.Request)
						seen[event.Request] = true
					}
					if event.Name == "l2_publish" && event.Destination != pool {
						t.Fatalf("injected target ignored: %+v", event)
					}
				}
				want := []string{"blocker", "long", "short"}
				if order == "sjf" {
					want = []string{"blocker", "short", "long"}
				}
				if len(scheduled) < 3 || !reflect.DeepEqual(scheduled[:3], want) {
					t.Fatalf("actual admission %v, want prefix %v", scheduled, want)
				}
				if req.calls == 0 || peer.calls == 0 || pools[pool].events == 0 || pools[pool].victims == 0 || r.Counts["l2_publish"] == 0 {
					t.Fatalf("missing policy execution: req=%d peer=%d pool=%+v", req.calls, peer.calls, pools[pool])
				}
				if len(r.FirstTokenUS) != len(c.Requests) || len(r.FinishedUS) != len(c.Requests) {
					t.Fatal("incomplete requests")
				}
				for _, state := range r.Pools {
					if state["reserved"] != 0 || state["read_pins"] != 0 {
						t.Fatalf("pool resources leaked: %v", state)
					}
				}
				if r.HBM["instance_0"]["active_or_pinned"] != 0 || len(r.InjectedPolicies) != 3 || r.KVPolicyCostCoverage == "" {
					t.Fatal("missing provenance/coverage or leaked HBM")
				}
			})
		}
	}
}

func TestPolicyFactoriesCreatePerInstanceAndPerPoolState(t *testing.T) {
	c := labConfig(2, "both")
	c.DecisionPolicy = &DecisionPolicyConfig{}
	reqs := map[string]*countedDecision{}
	peers := map[string]*selectedPoolPolicy{}
	pools := map[string]*countedPoolPolicy{}
	r, err := RunWithPolicies(c, PolicyFactories{
		Decision: func(id string) sim.DecisionPolicy {
			base, _ := sim.NewQueueDecisionPolicy("fcfs", 0)
			p := &countedDecision{base: base}
			if reqs[id] != nil {
				t.Fatalf("duplicate decision construction %s", id)
			}
			reqs[id] = p
			return p
		},
		Peer: func(id string) kv.PeerPolicy {
			p := &selectedPoolPolicy{pool: "dram"}
			if peers[id] != nil {
				t.Fatalf("duplicate peer construction %s", id)
			}
			peers[id] = p
			return p
		},
		PoolEviction: func(id string) kv.PoolEvictionPolicy {
			p := &countedPoolPolicy{}
			if pools[id] != nil {
				t.Fatalf("duplicate shared pool construction %s", id)
			}
			pools[id] = p
			return p
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 2 || len(peers) != 2 || len(pools) != 2 || len(r.FinishedUS) != len(c.Requests) {
		t.Fatal("factories have wrong topology scope")
	}
	for id, req := range reqs {
		if req.calls == 0 || peers[id] == nil || peers[id].calls == 0 {
			t.Fatalf("unused or mismatched instance policies: %s", id)
		}
	}
}

func TestPolicyFactoriesPreserveDefaultsAndDecisionWrapper(t *testing.T) {
	c := labConfig(1, "both")
	c.DecisionPolicy = &DecisionPolicyConfig{}
	baseline, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	empty, err := RunWithPolicies(c, PolicyFactories{})
	if err != nil || !reflect.DeepEqual(baseline.Events, empty.Events) || !reflect.DeepEqual(baseline.FirstTokenUS, empty.FirstTokenUS) || empty.InjectedPolicies != nil {
		t.Fatal("empty factories changed default execution or metadata", err)
	}
	factory := func(string) sim.DecisionPolicy {
		p, _ := sim.NewQueueDecisionPolicy("sjf", 0)
		return p
	}
	a, err := RunWithDecisionPolicy(c, factory)
	if err != nil {
		t.Fatal(err)
	}
	b, err := RunWithPolicies(c, PolicyFactories{Decision: factory})
	if err != nil || !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) || !reflect.DeepEqual(a.InjectedPolicies, b.InjectedPolicies) || b.KVPolicyCostCoverage != "" {
		t.Fatal("decision wrapper changed execution", err)
	}
}

func TestPolicyFactoriesRejectNilPoliciesBeforeExecution(t *testing.T) {
	for name, factories := range map[string]PolicyFactories{
		"decision":       {Decision: func(string) sim.DecisionPolicy { return nil }},
		"typed decision": {Decision: func(string) sim.DecisionPolicy { return (*countedDecision)(nil) }},
		"peer":           {Peer: func(string) kv.PeerPolicy { return nil }},
		"typed peer":     {Peer: func(string) kv.PeerPolicy { return (*selectedPoolPolicy)(nil) }},
		"pool":           {PoolEviction: func(string) kv.PoolEvictionPolicy { return nil }},
		"typed pool":     {PoolEviction: func(string) kv.PoolEvictionPolicy { return (*countedPoolPolicy)(nil) }},
	} {
		t.Run(name, func(t *testing.T) {
			c := labConfig(1, "both")
			c.DecisionPolicy = &DecisionPolicyConfig{}
			r, err := RunWithPolicies(c, factories)
			if r != nil || err == nil || !strings.Contains(err.Error(), "factory returned nil") {
				t.Fatal("invalid factory must fail before simulation", r, err)
			}
		})
	}
	c := labConfig(1, "both")
	called := false
	_, err := RunWithPolicies(c, PolicyFactories{Decision: func(string) sim.DecisionPolicy { called = true; return nil }})
	if err == nil || called {
		t.Fatal("missing decision runtime config must fail before constructing a policy")
	}
}
