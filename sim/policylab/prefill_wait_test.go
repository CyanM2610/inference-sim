package policylab

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestResidentPrefillWaitPublicFactoryControlsActualWork(t *testing.T) {
	c := labConfig(1, "hbm")
	c.Instances[0].HBMBlocks = 32
	c.MaxBatchTokens, c.PrefillChunk = 32, 16
	c.Requests = c.Requests[:2]
	c.Requests[0].Input = c.Requests[0].Input[:33]
	c.Requests[1].Input = c.Requests[1].Input[:17]
	for i := range c.Requests {
		c.Requests[i].At = 0
		c.Requests[i].Output = []sim.TokenID{9}
	}
	c.DecisionPolicy = &DecisionPolicyConfig{PrefillWaits: true, TraceEvents: true,
		ExtraCost: &sim.LinearDecisionCost{PerPrefillWaitUS: 7, Provenance: "synthetic wait sensitivity"}}
	results := map[bool]*Result{}
	for _, enabled := range []bool{false, true} {
		duration := int64(0)
		if enabled {
			duration = 100
		}
		c.DecisionPolicy.PrefillWaitProbe = &PrefillWaitProbeConfig{DurationUS: duration, Requests: []string{"r0_0"}}
		result, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.FinishedUS) != 2 || result.PrefillWaitCostCoverage == "" {
			t.Fatal("factory ignored wait capability or lost completion", enabled, result.FinishedUS)
		}
		var updates, grants int64
		var until int64
		for _, d := range result.PolicyDecisions {
			if len(d.Plan.Waits) > 0 {
				updates++
				until = d.Plan.Waits[0].UntilUS
				if d.Service == nil || d.Service.ExtraUS != 7 || len(d.Feedback.WaitUpdates) != 1 {
					t.Fatal("wait service/feedback not on execution path", d)
				}
			}
			for _, g := range d.Feedback.Grants {
				grants += g.Tokens
				if g.Request == "r0_0" && until > 0 && d.Feedback.AtUS < until {
					t.Fatal("resident request advanced before expiry", d)
				}
			}
		}
		if grants != 50 || enabled && updates != 1 || !enabled && updates != 0 {
			t.Fatal("request was recomputed, lost or never paused", grants, updates)
		}
		results[enabled] = result
	}
	if results[true].FirstTokenUS["r0_0"] <= results[false].FirstTokenUS["r0_0"] {
		t.Fatal("wait did not affect actual TTFT")
	}
}

func TestPrefillWaitProbeConfigSelectionRoundTripAndValidation(t *testing.T) {
	for _, selection := range []string{"", `,"requests":null`, `,"requests":[]`, `,"requests":["r0_0"]`} {
		c, err := Decode([]byte(`{"decision_policy":{"prefill_waits":true,"prefill_wait_probe":{"duration_us":100` + selection + `}}}`))
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(c)
		if err != nil {
			t.Fatal(err)
		}
		roundTrip, err := Decode(encoded)
		if err != nil || !reflect.DeepEqual(c.DecisionPolicy.PrefillWaitProbe, roundTrip.DecisionPolicy.PrefillWaitProbe) {
			t.Fatal("request selection changed during config round trip", string(encoded), err)
		}
		base, _ := sim.NewQueueDecisionPolicy("fcfs", 0)
		probe, err := roundTrip.DecisionPolicy.PrefillWaitProbe.NewPolicy(base)
		if err != nil {
			t.Fatal(err)
		}
		plan := probe.Decide(sim.DecisionView{NowUS: 10, Capabilities: sim.DecisionCapabilities{PrefillWaits: true},
			Running: []sim.DecisionRequest{{ID: "r0_0", PrefillWaitable: true}, {ID: "r0_1", PrefillWaitable: true}}})
		want := 2
		if selection == `,"requests":[]` {
			want = 0
		} else if selection == `,"requests":["r0_0"]` {
			want = 1
		}
		if len(plan.Waits) != want {
			t.Fatal("config changed selection meaning", selection, plan.Waits)
		}
	}
	c := labConfig(1, "hbm")
	c.DecisionPolicy = &DecisionPolicyConfig{PrefillWaitProbe: &PrefillWaitProbeConfig{}}
	if _, err := Run(c); err == nil || !strings.Contains(err.Error(), "requires prefill_waits") {
		t.Fatal("probe accepted missing capability even at zero duration", err)
	}
	c.DecisionPolicy.PrefillWaits = true
	c.DecisionPolicy.PrefillWaitProbe.DurationUS = -1
	if _, err := Run(c); err == nil || !strings.Contains(err.Error(), "nonnegative") {
		t.Fatal("probe accepted negative duration", err)
	}
}

func TestPrefillWaitProbeDecoratesCustomFactory(t *testing.T) {
	c := decisionLabConfig("hbm", "fcfs")
	c.Requests = c.Requests[:1]
	c.Requests[0].Output = []sim.TokenID{9}
	c.DecisionPolicy = &DecisionPolicyConfig{PrefillWaits: true,
		PrefillWaitProbe: &PrefillWaitProbeConfig{DurationUS: 100}}
	var base *feedbackCapPolicy
	r, err := RunWithDecisionPolicy(c, func(string) sim.DecisionPolicy {
		base = &feedbackCapPolicy{}
		return base
	})
	if err != nil {
		t.Fatal(err)
	}
	updates := 0
	for _, d := range r.PolicyDecisions {
		updates += len(d.Feedback.WaitUpdates)
	}
	if updates != 1 || base.rounds < 2 || len(r.FirstTokenUS) != 1 {
		t.Fatal("configured probe bypassed custom base or lost feedback", updates, base.rounds)
	}
}

func TestPrefillWaitProbeJSONRejectsInvalidDurationAndSelection(t *testing.T) {
	for _, value := range []string{
		`{}`, `{"duration_us":null}`, `{"duration_us":false}`, `{"duration_us":1.5}`,
		`{"duration_us":-1}`, `{"duration_us":50,"requests":[null]}`,
		`{"duration_us":50,"requests":[1]}`, `{"duration_us":50,"requests":{}}`,
		`{"duration_us":50,"unknown":true}`,
	} {
		var probe PrefillWaitProbeConfig
		if err := json.Unmarshal([]byte(value), &probe); err == nil {
			t.Fatal("accepted invalid Python probe configuration", value)
		}
	}
}

func TestPrefillWaitProbeJSONProgressThreshold(t *testing.T) {
	for _, raw := range []string{
		`{"duration_us":10,"min_computed_tokens":null}`,
		`{"duration_us":10,"min_computed_tokens":-1}`,
		`{"duration_us":10,"min_computed_tokens":false}`,
		`{"duration_us":10,"min_computed_tokens":1.5}`,
		`{"duration_us":10,"min_computed_tokens":"32"}`,
	} {
		var probe PrefillWaitProbeConfig
		if err := json.Unmarshal([]byte(raw), &probe); err == nil {
			t.Fatal("invalid threshold accepted", raw)
		}
	}
	var probe PrefillWaitProbeConfig
	if err := json.Unmarshal([]byte(`{"duration_us":10,"min_computed_tokens":32}`), &probe); err != nil {
		t.Fatal(err)
	}
	if probe.MinComputedTokens != 32 {
		t.Fatal("threshold lost", probe)
	}
	encoded, err := json.Marshal(probe)
	if err != nil {
		t.Fatal(err)
	}
	var decoded PrefillWaitProbeConfig
	if err := json.Unmarshal(encoded, &decoded); err != nil || decoded.MinComputedTokens != 32 {
		t.Fatal("threshold roundtrip", decoded, err)
	}
}

func TestPrefillWaitProbeNoOpPreservesExecutionAndBaseEvents(t *testing.T) {
	c := decisionLabConfig("hbm", "fcfs")
	c.Requests = c.Requests[:2]
	c.Requests[1].At = 5
	c.DecisionPolicy = &DecisionPolicyConfig{PrefillWaits: true,
		ExtraCost: &sim.LinearDecisionCost{FixedUS: 10, Provenance: "synthetic callback fixture"}}
	var base *arrivalFeedbackPolicy
	factory := func(string) sim.DecisionPolicy {
		base = &arrivalFeedbackPolicy{}
		return base
	}
	before, err := RunWithDecisionPolicy(c, factory)
	if err != nil {
		t.Fatal(err)
	}
	for _, probe := range []*PrefillWaitProbeConfig{{DurationUS: 0}, {DurationUS: 100, Requests: []string{}}} {
		c.DecisionPolicy.PrefillWaitProbe = probe
		after, err := RunWithDecisionPolicy(c, factory)
		if err != nil {
			t.Fatal(err)
		}
		if base.arrivals != 2 || len(after.PolicyEvents) != len(before.PolicyEvents) {
			t.Fatal("probe lost the base policy's lifecycle subscription", base.arrivals)
		}
		if !reflect.DeepEqual(before.Events, after.Events) || !reflect.DeepEqual(before.FirstTokenUS, after.FirstTokenUS) ||
			!reflect.DeepEqual(before.FinishedUS, after.FinishedUS) || !reflect.DeepEqual(before.BatchShapes, after.BatchShapes) ||
			!reflect.DeepEqual(before.Requests, after.Requests) || len(before.PolicyDecisions) != len(after.PolicyDecisions) {
			t.Fatal("inactive probe changed execution")
		}
		for i, d := range before.PolicyDecisions {
			if !reflect.DeepEqual(d.Plan, after.PolicyDecisions[i].Plan) || !reflect.DeepEqual(d.Feedback, after.PolicyDecisions[i].Feedback) {
				t.Fatal("inactive probe changed a base decision or its outcome")
			}
		}
	}
}

func TestResidentPrefillWaitRejectsUncoveredExecutionProfiles(t *testing.T) {
	c := labConfig(1, "hbm")
	c.DecisionPolicy = &DecisionPolicyConfig{PrefillWaits: true, HostServiceCost: &CapacityHostServiceCostConfig{}}
	if _, err := Run(c); err == nil || !strings.Contains(err.Error(), "outside existing") {
		t.Fatal("existing calibrated profile accepted new waiting path", err)
	}
}
