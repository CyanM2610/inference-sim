package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func decisionLabConfig(topology, order string) Config {
	c := labConfig(1, topology)
	c.RequestScheduler = order
	c.TraceBatchShapes = true
	if topology == "dram" {
		phaseTestConfig(&c)
	} else {
		c.BatchCost = &BatchCostConfig{CoefficientsUS: []float64{100, 1, 0, 1}, Provenance: "synthetic decision behavior fixture"}
	}
	for i := range c.Requests {
		c.Requests[i].At = 0
		if i%2 == 0 {
			c.Requests[i].Input = c.Requests[i].Input[:33]
		}
	}
	return c
}

func TestDecisionAdaptersPreserveFourBaselineBehaviors(t *testing.T) {
	for _, top := range []string{"hbm", "dram"} {
		for _, order := range []string{"fcfs", "sjf"} {
			t.Run(top+"/"+order, func(t *testing.T) {
				c := decisionLabConfig(top, order)
				before, err := Run(c)
				if err != nil {
					t.Fatal(err)
				}
				c.DecisionPolicy = &DecisionPolicyConfig{}
				after, err := Run(c)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(before.Events, after.Events) || !reflect.DeepEqual(before.FirstTokenUS, after.FirstTokenUS) || !reflect.DeepEqual(before.FinishedUS, after.FinishedUS) || !reflect.DeepEqual(before.BatchShapes, after.BatchShapes) || !reflect.DeepEqual(before.Requests, after.Requests) {
					t.Fatal("default decision adapter changed baseline execution")
				}
				if len(after.PolicyDecisions) == 0 || after.DecisionCostCoverage != "host_measured_sim_time_uncalibrated" {
					t.Fatal("decision evidence/cost coverage missing")
				}
			})
		}
	}
}

func TestDecisionTokenCapChangesRunningPrefillAndPreservesCompletion(t *testing.T) {
	c := decisionLabConfig("hbm", "sjf")
	before, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy = &DecisionPolicyConfig{QueueOrder: "sjf", PrefillTokenCap: 7}
	c.Requests = append(c.Requests, RequestConfig{ID: "future", At: 100000, Input: []sim.TokenID{99, 98, 97}, Output: []sim.TokenID{1}, MaxOutputTokens: 10, TTFTSLOUS: 2000})
	after, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(after.FirstTokenUS) != len(c.Requests) || len(after.BatchShapes) <= len(before.BatchShapes) {
		t.Fatal("token cap did not affect real batch execution")
	}
	var cappedRunning bool
	for _, record := range after.PolicyDecisions {
		caps := map[string]int64{}
		for _, c := range record.Plan.TokenCaps {
			caps[c.Request] = c.Tokens
		}
		for _, grant := range record.Feedback.Grants {
			if cap, ok := caps[grant.Request]; ok && grant.Tokens > cap {
				t.Fatal("runtime exceeded policy cap")
			}
		}
		for _, r := range record.View.Running {
			if caps[r.ID] > 0 && r.ComputedTokens > 0 && r.ComputedTokens < r.InputTokens {
				cappedRunning = true
			}
		}
		for _, r := range append(record.View.Waiting, record.View.Running...) {
			if r.ArrivalUS > record.View.NowUS {
				t.Fatal("future request leaked into decision")
			}
			if r.ID == "future" && (r.ClientOutputLimit != 10 || r.TTFTTargetUS != 2000) {
				t.Fatal("explicit client metadata not exposed")
			}
		}
	}
	if !cappedRunning {
		t.Fatal("fixture did not exercise running prefill cap")
	}
}

type feedbackCapPolicy struct{ rounds int }

func (p *feedbackCapPolicy) Decide(v sim.DecisionView) sim.DecisionPlan {
	cap := int64(1)
	if p.rounds > 0 {
		cap = 4
	}
	base, _ := sim.NewQueueDecisionPolicy("sjf", cap)
	return base.Decide(v)
}
func (p *feedbackCapPolicy) Observe(f sim.DecisionFeedback) {
	if f.Status == "applied" && len(f.Grants) > 0 {
		p.rounds++
	}
}

func TestStatefulDecisionPluginRunsThroughSharedRuntime(t *testing.T) {
	c := decisionLabConfig("hbm", "fcfs")
	c.DecisionPolicy = &DecisionPolicyConfig{}
	var policy *feedbackCapPolicy
	r, err := RunWithDecisionPolicy(c, func(string) sim.DecisionPolicy { policy = &feedbackCapPolicy{}; return policy })
	if err != nil {
		t.Fatal(err)
	}
	if len(r.FirstTokenUS) != len(c.Requests) || policy.rounds < 2 {
		t.Fatal("plugin did not receive execution feedback")
	}
	if r.PolicyDecisions[0].Plan.TokenCaps[0].Tokens != 1 || r.PolicyDecisions[1].Plan.TokenCaps[0].Tokens != 4 {
		t.Fatal("feedback did not influence next decision")
	}
}

func TestDecisionServiceZeroCostPreservesFourBaselineBehaviors(t *testing.T) {
	for _, top := range []string{"hbm", "dram"} {
		for _, order := range []string{"fcfs", "sjf"} {
			t.Run(top+"/"+order, func(t *testing.T) {
				c := decisionLabConfig(top, order)
				c.DecisionPolicy = &DecisionPolicyConfig{}
				before, err := Run(c)
				if err != nil {
					t.Fatal(err)
				}
				c.DecisionPolicy.ExtraCost = &sim.LinearDecisionCost{Provenance: "zero cost behavior test"}
				after, err := Run(c)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(before.Events, after.Events) || !reflect.DeepEqual(before.Requests, after.Requests) || !reflect.DeepEqual(before.FirstTokenUS, after.FirstTokenUS) || !reflect.DeepEqual(before.FinishedUS, after.FinishedUS) || !reflect.DeepEqual(before.BatchShapes, after.BatchShapes) {
					t.Fatal("zero cost changed baseline execution")
				}
				if after.DecisionCostCoverage != sim.ExplicitDecisionCostCoverage || len(after.PolicyDecisions) != len(before.PolicyDecisions) {
					t.Fatal("missing cost coverage or changed decision count")
				}
				for _, r := range after.PolicyDecisions {
					if r.Service == nil || r.Service.ExtraUS != 0 || r.Service.StartUS != r.Service.EndUS {
						t.Fatal("missing explicit zero cost evidence")
					}
				}
			})
		}
	}
}

func TestDecisionServiceConfigChargesPerDecisionOnRealRunner(t *testing.T) {
	c := decisionLabConfig("hbm", "fcfs")
	c.Requests = c.Requests[:1]
	c.Requests[0].Output = []sim.TokenID{7}
	c.DecisionPolicy = &DecisionPolicyConfig{}
	before, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.DecisionPolicy.ExtraCost = &sim.LinearDecisionCost{FixedUS: 10, PerVisibleRequestUS: 3, Provenance: "synthetic sensitivity test"}
	after, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(after.PolicyDecisions) != 1 || after.FirstTokenUS[c.Requests[0].ID]-before.FirstTokenUS[c.Requests[0].ID] != 13 {
		t.Fatal("configured service not charged in real runner")
	}
	r := after.PolicyDecisions[0]
	if r.Service == nil || r.Service.ExtraUS != 13 || r.Feedback.AtUS != r.View.NowUS+13 {
		t.Fatal("incorrect service trace")
	}
}

func TestDecisionEventTracePreservesFourBaselineBehaviors(t *testing.T) {
	for _, top := range []string{"hbm", "dram"} {
		for _, order := range []string{"fcfs", "sjf"} {
			t.Run(top+"/"+order, func(t *testing.T) {
				c := decisionLabConfig(top, order)
				c.DecisionPolicy = &DecisionPolicyConfig{}
				before, err := Run(c)
				if err != nil {
					t.Fatal(err)
				}
				c.DecisionPolicy.TraceEvents = true
				after, err := Run(c)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(before.Events, after.Events) || !reflect.DeepEqual(before.FirstTokenUS, after.FirstTokenUS) || !reflect.DeepEqual(before.FinishedUS, after.FinishedUS) || !reflect.DeepEqual(before.BatchShapes, after.BatchShapes) || !reflect.DeepEqual(before.Requests, after.Requests) {
					t.Fatal("event trace changed execution")
				}
				counts := map[string]int{}
				for _, r := range after.PolicyEvents {
					counts[r.Event.Kind]++
					if r.CallbackWallNS != 0 || r.CostCoverage != sim.DecisionEventCostCoverage {
						t.Fatal("trace-only path claimed callback/calibration")
					}
				}
				for _, kind := range []string{"request_arrived", "first_output", "request_completed"} {
					if counts[kind] != len(c.Requests) {
						t.Fatalf("%s count=%d", kind, counts[kind])
					}
				}
				if counts["batch_returned"] != len(after.BatchShapes) {
					t.Fatal("batch notifications lost")
				}
				if top == "dram" && counts["transfer_adopted"] == 0 {
					t.Fatal("transfer notifications lost")
				}
			})
		}
	}
}

type arrivalFeedbackPolicy struct{ arrivals int }

func (p *arrivalFeedbackPolicy) Decide(v sim.DecisionView) sim.DecisionPlan {
	cap := int64(1)
	if p.arrivals > 1 {
		cap = 4
	}
	base, _ := sim.NewQueueDecisionPolicy("fcfs", cap)
	return base.Decide(v)
}
func (p *arrivalFeedbackPolicy) Observe(sim.DecisionFeedback) {}
func (p *arrivalFeedbackPolicy) OnEvent(e sim.DecisionEvent) {
	if e.Kind == "request_arrived" {
		p.arrivals++
	}
}

func TestDecisionEventListenerRunsThroughFactoryWithoutCoreChanges(t *testing.T) {
	c := decisionLabConfig("hbm", "fcfs")
	c.Requests = c.Requests[:2]
	c.Requests[1].At = 5
	c.DecisionPolicy = &DecisionPolicyConfig{ExtraCost: &sim.LinearDecisionCost{FixedUS: 10, Provenance: "synthetic callback fixture"}}
	var p *arrivalFeedbackPolicy
	r, err := RunWithDecisionPolicy(c, func(string) sim.DecisionPolicy { p = &arrivalFeedbackPolicy{}; return p })
	if err != nil {
		t.Fatal(err)
	}
	if p.arrivals != 2 || r.PolicyDecisions[0].Plan.TokenCaps[0].Tokens != 1 || r.PolicyDecisions[1].Plan.TokenCaps[0].Tokens != 4 || len(r.PolicyEvents) == 0 || len(r.FirstTokenUS) != 2 {
		t.Fatal("factory did not forward stateful lifecycle notifications")
	}
}

func TestDecisionWaitConfigGatesActualAdmission(t *testing.T) {
	c := decisionLabConfig("hbm", "fcfs")
	c.Requests = c.Requests[:1]
	c.Requests[0].Output = []sim.TokenID{9}
	c.DecisionPolicy = &DecisionPolicyConfig{TraceEvents: true, AdmissionDelayUS: 1000}
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.PolicyDecisions) != 2 || len(r.PolicyDecisions[0].Feedback.Grants) != 0 || r.PolicyDecisions[0].Feedback.Unselected[0].Reason != "policy_wait" || r.PolicyDecisions[1].Feedback.AtUS != 1000 || r.FirstTokenUS[c.Requests[0].ID] <= 1000 {
		t.Fatal("JSON wait did not gate execution")
	}
	if r.Counts["sim.DecisionTimerEvent"] != 1 || r.PolicyDecisions[0].Feedback.WaitUpdates[0].UntilUS != 1000 {
		t.Fatal("missing timer/action evidence")
	}
}

type revisitPolicy struct{}

func (*revisitPolicy) Decide(v sim.DecisionView) sim.DecisionPlan {
	at := int64(1000000)
	return sim.DecisionPlan{Version: v.Version, RevisitAtUS: &at}
}
func (*revisitPolicy) Observe(sim.DecisionFeedback) {}

func TestDecisionRevisitDoesNotInflateClusterEndAfterCompletion(t *testing.T) {
	c := decisionLabConfig("hbm", "fcfs")
	c.Requests = c.Requests[:1]
	c.Requests[0].Output = []sim.TokenID{9}
	c.DecisionPolicy = &DecisionPolicyConfig{}
	r, err := RunWithDecisionPolicy(c, func(string) sim.DecisionPolicy { return &revisitPolicy{} })
	if err != nil {
		t.Fatal(err)
	}
	if r.Counts["sim.DecisionTimerEvent"] != 0 {
		t.Fatal("orphan timer executed after completion")
	}
	for _, e := range r.Events {
		if e.Time >= 1000000 {
			t.Fatal("orphan timer advanced cluster time")
		}
	}
}
