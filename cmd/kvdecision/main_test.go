package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/policylab"
)

func residentView() sim.DecisionView {
	return sim.DecisionView{Version: 1, NowUS: 100, MaxBatchTokens: 32, MaxSequences: 2, BlockTokens: 16,
		Running:      []sim.DecisionRequest{{ID: "long", State: sim.RequestState("running"), InputTokens: 128, ComputedTokens: 16, PrefillWaitable: true}},
		Capabilities: sim.DecisionCapabilities{QueueOrder: true, WaitUntil: true, PrefillWaits: true}}
}

func queueOracleInput() input {
	curve := policylab.EnginePhaseCurve{PreForward: []float64{1, 0, 0, 0}, PostForward: []float64{10, 1, 0, 0},
		GPUReady: []float64{10, 1, 0, 0}, OutputReady: []float64{10, 1, 0, 0}, Poll: []float64{1, 0, 0, 0}, Tail: []float64{1, 0, 0, 0}}
	c := policylab.Config{RestoreControl: true, DecisionEstimates: &policylab.DecisionEstimateConfig{MaxDecodeContextTokens: 2048},
		Mechanisms: &kv.PeerMechanisms{RestoreWindow: 1},
		Instances:  []policylab.InstanceConfig{{Access: []kv.PeerAccess{{Pool: "dram", ReadPath: []string{"copy"}, WritePath: []string{"copy"}}}}},
		Resources:  []kv.PeerResource{{ID: "copy", BytesPerUS: 1000}},
		EnginePhases: &policylab.EnginePhaseCostConfig{Compute: curve, BlockBytes: 16,
			LoadSubmit: []float64{1, 1}, StoreSubmit: []float64{1, 1}, Provenance: "synthetic oracle fixture"},
		DecisionPolicy: &policylab.DecisionPolicyConfig{BudgetRestore: &policylab.BudgetRestoreConfig{Mode: "budget_queues", Guardband: 1, PrimaryWeight: 3, BestEffortWeight: 1}}}
	v := sim.DecisionView{Version: 1, BlockTokens: 16, MaxBatchTokens: 7, MaxSequences: 2, PrefillChunk: 7,
		Capabilities: sim.DecisionCapabilities{QueueOrder: true, TokenCaps: true, AdmissionSelection: true, RestoreChoice: true,
			CostEstimates: true, CapacityPreemption: true, PrefillPreemption: true},
		KV: sim.DecisionKVState{PrefixStateKnown: true, TransfersKnown: true}}
	for i, id := range []string{"primary", "best"} {
		slo := int64(1000000000)
		if i == 1 {
			slo = 1
		}
		v.Waiting = append(v.Waiting, sim.DecisionRequest{ID: id, State: sim.StateQueued, InputTokens: 32, TTFTTargetUS: slo, ClientOutputLimit: 3})
		v.KV.Requests = append(v.KV.Requests, sim.DecisionKVRequest{ID: id})
	}
	return input{Mode: "budget_queues", Guardband: 1, Config: c, View: v}
}

func TestOracleBudgetQueuesRequireAndUseActualFeedback(t *testing.T) {
	in := queueOracleInput()
	first, err := evaluate(in)
	if err != nil {
		t.Fatal(err)
	}
	if first.QueueService == nil || first.QueueService.PrimaryTokens != 0 {
		t.Fatal("new proposal counted as service")
	}
	previous := in.View
	in.View.Version, in.View.NowUS = 2, 1
	in.Replay = []decisionReplay{{View: previous, Feedback: sim.DecisionFeedback{Version: 1, AtUS: 0, Status: "applied",
		Grants: []sim.DecisionGrant{{Request: "primary", Tokens: 3}}}}}
	later, err := evaluate(in)
	if err != nil {
		t.Fatal(err)
	}
	if later.QueueService.PrimaryTokens != 3 || later.QueueService.BestEffortTokens != 0 || later.QueueService.DeficitUnits != 3 {
		t.Fatal("replay did not retain actual service", later.QueueService)
	}
	cap := func(p sim.DecisionPlan, id string) int64 {
		for _, c := range p.TokenCaps {
			if c.Request == id {
				return c.Tokens
			}
		}
		return 0
	}
	if cap(later.Plan, "best") <= cap(first.Plan, "best") {
		t.Fatal("feedback did not change policy decision")
	}
	for _, bad := range []input{
		{Mode: "budget_queues", ViewHistory: []sim.DecisionView{previous}},
		{Mode: "budget_queues", History: []sim.DecisionFeedback{{Status: "applied"}}},
	} {
		if _, err := evaluate(bad); err == nil {
			t.Fatal("missing decision/feedback ordering accepted")
		}
	}
	in.Replay[0].Feedback.AtUS = 2
	if _, err := evaluate(in); err == nil {
		t.Fatal("future feedback was consumed")
	}
	in.Replay[0].Feedback.AtUS = 0
	in.Replay[0].Feedback.Grants[0].Tokens = 7
	if _, err := evaluate(in); err == nil {
		t.Fatal("grants beyond prior quota accepted")
	}
}

func TestOracleCurrentFeedbackDoesNotInfluenceCurrentPlan(t *testing.T) {
	in := queueOracleInput()
	before, err := evaluate(in)
	if err != nil {
		t.Fatal(err)
	}
	in.Feedback = &sim.DecisionFeedback{Version: in.View.Version, Status: "applied", Grants: []sim.DecisionGrant{{Request: "primary", Tokens: 3}}}
	after, err := evaluate(in)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(before.Plan)
	b, _ := json.Marshal(after.Plan)
	if !bytes.Equal(a, b) || !after.QueueServiceAfterFeedback || after.QueueService.PrimaryTokens != 3 || before.QueueService.PrimaryTokens != 0 {
		t.Fatal("current feedback leaked into decision or was not applied after it")
	}
}

func TestOraclePrefillWaitHistoryPreservesOneShotState(t *testing.T) {
	in := input{Mode: "demand_fcfs", View: residentView(), PrefillWaitProbe: &policylab.PrefillWaitProbeConfig{DurationUS: 50}}
	first, err := evaluate(in)
	if err != nil || len(first.Plan.Waits) != 1 || first.Plan.Waits[0].UntilUS != 150 {
		t.Fatal("resident view was not evaluated", first, err)
	}
	in.ViewHistory = []sim.DecisionView{in.View}
	in.View.Version, in.View.NowUS, in.View.Running[0].WaitUntilUS = 2, 110, 150
	later, err := evaluate(in)
	if err != nil || len(later.Plan.Waits) != 0 {
		t.Fatal("history reissued resident wait", later, err)
	}
	in.ViewHistory = nil
	independent, err := evaluate(in)
	if err != nil || len(independent.Plan.Waits) != 1 {
		t.Fatal("oracle line did not construct independent state", independent, err)
	}
}

func TestOracleResidentViewsRemainStrictAndRequireCapability(t *testing.T) {
	in := input{Mode: "demand_fcfs", View: residentView(), PrefillWaitProbe: &policylab.PrefillWaitProbeConfig{DurationUS: 50}}
	data, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var result bytes.Buffer
	if err := runStreams(bytes.NewReader(data), &result); err != nil || !strings.Contains(result.String(), "until_us") {
		t.Fatal("current JSON schema rejected actual resident fields", result.String(), err)
	}
	bad := strings.Replace(string(data), `"prefill_waitable":true`, `"prefill_waitable":true,"unknown_wait_field":true`, 1)
	if err := runStreams(strings.NewReader(bad), &result); err == nil || !strings.Contains(err.Error(), "unknown_wait_field") {
		t.Fatal("oracle silently ignored an unknown native view field", err)
	}
	in.View.Capabilities.PrefillWaits = false
	if _, err := evaluate(in); err == nil || !strings.Contains(err.Error(), "capability") {
		t.Fatal("probe accepted capability mismatch", err)
	}
	in.View = residentView()
	in.ViewHistory, in.History = []sim.DecisionView{in.View}, []sim.DecisionFeedback{{Status: "applied"}}
	if _, err := evaluate(in); err == nil || !strings.Contains(err.Error(), "ordering") {
		t.Fatal("oracle claimed ambiguous feedback replay", err)
	}
}

func TestOracleProbeConfigurationPreservesEmptySelection(t *testing.T) {
	for _, selection := range [][]string{nil, {}, {"long"}, {"other"}} {
		in := input{Mode: "demand_fcfs", View: residentView(), Config: policylab.Config{DecisionPolicy: &policylab.DecisionPolicyConfig{
			PrefillWaits: true, PrefillWaitProbe: &policylab.PrefillWaitProbeConfig{DurationUS: 50, Requests: selection}}}}
		want := selection == nil || len(selection) > 0 && selection[0] == "long"
		out, err := evaluate(in)
		if err != nil || (len(out.Plan.Waits) > 0) != want {
			t.Fatal(selection, out, err)
		}
		in.PrefillWaitProbe = in.Config.DecisionPolicy.PrefillWaitProbe
		if _, err := evaluate(in); err == nil {
			t.Fatal("accepted duplicate probe configuration")
		}
	}
}
