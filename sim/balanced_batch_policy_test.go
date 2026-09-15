package sim

import (
	"encoding/json"
	"reflect"
	"testing"
)

func balancedView() DecisionView {
	v := DecisionView{Version: 1, NowUS: 100, BlockTokens: 16, MaxBatchTokens: 512, MaxSequences: 4, PrefillChunk: 128,
		Capabilities: DecisionCapabilities{AdmissionSelection: true, TokenCaps: true, RestoreChoice: true, QueueOrder: true},
		KV:           DecisionKVState{PrefixStateKnown: true}}
	for _, id := range []string{"a-hot", "b-hot", "c-cold"} {
		v.Waiting = append(v.Waiting, DecisionRequest{ID: id, InputTokens: 256})
		q := DecisionKVRequest{ID: id}
		if id != "c-cold" {
			q.RecoverablePrefixBlocks = 16
			q.RecoverablePrefixTokens = 255
		}
		v.KV.Requests = append(v.KV.Requests, q)
	}
	return v
}

func TestBalancedBatchSelectsComputePartnerAndKeepsOldest(t *testing.T) {
	v := balancedView()
	p, _ := NewBalancedBatchPolicy(100, 128, 2, 0)
	before := cloneDecisionView(v)
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.Admission.Requests, []string{"a-hot", "c-cold"}) || !reflect.DeepEqual(plan.QueueOrder, []string{"a-hot", "c-cold", "b-hot"}) {
		t.Fatal("no balanced selection", plan)
	}
	if !reflect.DeepEqual(plan.TokenCaps, []DecisionTokenCap{{"a-hot", 1}, {"c-cold", 127}}) {
		t.Fatal("quota does not respect remaining batch room", plan.TokenCaps)
	}
	if !reflect.DeepEqual(v, before) {
		t.Fatal("policy changed the view")
	}
	p.MaxRequests = 3
	if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"a-hot", "c-cold"}) {
		t.Fatal("exhausted budget admitted another request", got)
	}
	p.TokenBudget = 512
	if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"a-hot", "c-cold", "b-hot"}) {
		t.Fatal("deprioritized request not refilled", got)
	}
}

func TestBalancedBatchRunningQuotasReserveProgressAndSkipPending(t *testing.T) {
	v := balancedView()
	v.Running = []DecisionRequest{{ID: "r0", InputTokens: 256}, {ID: "r1", InputTokens: 256}, {ID: "decode", InputTokens: 16, ComputedTokens: 17}}
	v.KV.Requests[0].TransferPending = true
	p, _ := NewBalancedBatchPolicy(100, 64, 0, 0)
	plan := p.Decide(v)
	if !reflect.DeepEqual(plan.TokenCaps, []DecisionTokenCap{{"r0", 62}, {"r1", 1}, {"decode", 1}}) || len(plan.Admission.Requests) != 0 {
		t.Fatal("running quota escaped aggregate budget", plan)
	}
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	p.TokenBudget = 512
	plan = p.Decide(v)
	if len(plan.Admission.Requests) != 1 || plan.Admission.Requests[0] != "b-hot" {
		t.Fatal("pending request selected, or oldest eligible not selected", plan)
	}
}

func TestAdmissionPlanCopiesAndRejectsInvalidSubsets(t *testing.T) {
	v := balancedView()
	for _, ids := range [][]string{{"future"}, {"a-hot", "a-hot"}} {
		if ValidateDecision(v, DecisionPlan{Version: v.Version, Admission: &DecisionAdmission{Requests: ids}}) == nil {
			t.Fatal("invalid admission accepted", ids)
		}
	}
	p := DecisionPlan{Version: v.Version, Admission: &DecisionAdmission{Requests: []string{}}}
	cloned := cloneDecisionPlan(p)
	raw, _ := json.Marshal(cloned)
	var restored DecisionPlan
	if err := json.Unmarshal(raw, &restored); err != nil || restored.Admission == nil || restored.Admission.Requests == nil {
		t.Fatal("empty admission lost across trace", string(raw), err)
	}
	p.Admission.Requests = []string{"a-hot"}
	cloned = cloneDecisionPlan(p)
	cloned.Admission.Requests[0] = "b-hot"
	if p.Admission.Requests[0] != "a-hot" {
		t.Fatal("admission aliases caller memory")
	}
}
