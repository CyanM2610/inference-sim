package main

import (
	"encoding/json"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/policylab"
)

func deferralInput() input {
	v := residentView()
	v.Capabilities.PrefillDeferrals = true
	v.Running[0].PrefillDeferrable = true
	v.Running = append(v.Running, sim.DecisionRequest{ID: "decode", State: sim.StateRunning, InputTokens: 17, ComputedTokens: 18, EmittedTokens: 2})
	return input{Mode: "demand_fcfs", View: v, Config: policylab.Config{DecisionPolicy: &policylab.DecisionPolicyConfig{
		PrefillDeferrals: true, PrefillDeferralProbe: &policylab.PrefillDeferralProbeConfig{MaxRounds: 1}}}}
}

func TestDeferralOracleRoundTripAndActualFeedback(t *testing.T) {
	for _, selection := range [][]string{nil, {}, {"long"}, {"other"}} {
		in := deferralInput()
		in.Config.DecisionPolicy.PrefillDeferralProbe.Requests = selection
		data, err := json.Marshal(in)
		if err != nil {
			t.Fatal(err)
		}
		var decoded input
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatal(err)
		}
		out, err := evaluate(decoded)
		want := selection == nil || len(selection) > 0 && selection[0] == "long"
		if err != nil || (len(out.Plan.PrefillDeferrals) > 0) != want {
			t.Fatal(selection, out, err)
		}
	}
	for _, reason := range []string{"policy_prefill_deferred", "preempted"} {
		in := deferralInput()
		previous := in.View
		in.View.Version, in.View.NowUS = 2, 110
		in.Replay = []decisionReplay{{View: previous, Feedback: sim.DecisionFeedback{Version: 1, AtUS: 100, Status: "applied",
			Grants: []sim.DecisionGrant{{Request: "decode", Tokens: 1}}, Unselected: []sim.DecisionUnselected{{Request: "long", Reason: reason}}}}}
		out, err := evaluate(in)
		if err != nil || (len(out.Plan.PrefillDeferrals) > 0) != (reason == "preempted") {
			t.Fatal(reason, out, err)
		}
	}
}

func TestDeferralOracleRejectsContradictoryActualFeedback(t *testing.T) {
	in := deferralInput()
	out, err := evaluate(in)
	if err != nil {
		t.Fatal(err)
	}
	for _, feedback := range []sim.DecisionFeedback{
		{Grants: []sim.DecisionGrant{{Request: "long", Tokens: 1}}},
		{Grants: []sim.DecisionGrant{{Request: "decode", Tokens: 1}}},
		{Unselected: []sim.DecisionUnselected{{Request: "long"}, {Request: "long"}}},
		{Unselected: []sim.DecisionUnselected{{Request: "long"}, {Request: "unknown"}}},
	} {
		feedback.Version, feedback.AtUS, feedback.Status = 1, 100, "applied"
		if err := validateReplayFeedback(in.View, out.Plan, feedback); err == nil {
			t.Fatal("invalid feedback accepted", feedback)
		}
	}
}
