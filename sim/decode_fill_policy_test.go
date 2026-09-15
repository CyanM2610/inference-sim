package sim

import (
	"reflect"
	"testing"
)

func decodeFillView() DecisionView {
	v := DecisionView{Version: 1, NowUS: 100, BlockTokens: 16, MaxBatchTokens: 32, MaxSequences: 4, PrefillChunk: 8, HBMFreeBlocks: 32,
		Capabilities: DecisionCapabilities{QueueOrder: true, TokenCaps: true, AdmissionSelection: true, RestoreChoice: true, Promotions: true, PromotionRetention: "request", PrefillDeferrals: true},
		KV:           DecisionKVState{PrefixStateKnown: true, TransfersKnown: true}}
	v.Running = []DecisionRequest{{ID: "decode", State: StateRunning, InputTokens: 17, ComputedTokens: 18, EmittedTokens: 2},
		{ID: "resident", State: StateRunning, InputTokens: 128, ComputedTokens: 16, PrefillDeferrable: true}}
	v.Waiting = []DecisionRequest{{ID: "hot", State: StateQueued, InputTokens: 129}, {ID: "cold", State: StateQueued, InputTokens: 65, ArrivalUS: 1}}
	v.KV.Requests = []DecisionKVRequest{{ID: "decode"}, {ID: "resident"}, {ID: "hot", RecoverablePrefixBlocks: 8, RecoverablePrefixTokens: 128}, {ID: "cold"}}
	return v
}

func decodeFillTestPolicy(t *testing.T) *DecodeFillPolicy {
	t.Helper()
	base, err := NewBalancedBatchPolicy(1, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewDecodeFillPolicy(base)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func decodeFillDecide(t *testing.T, p *DecodeFillPolicy, v DecisionView) DecisionPlan {
	t.Helper()
	before := cloneDecisionView(v)
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err, plan)
	}
	if !reflect.DeepEqual(v, before) {
		t.Fatal("policy mutated snapshot")
	}
	return plan
}

func TestDecodeFillWaitsForActualAdoptionAndRetainsSelection(t *testing.T) {
	v := decodeFillView()
	p := decodeFillTestPolicy(t)
	first := decodeFillDecide(t, p, v)
	if len(first.Promotions) != 1 || first.Promotions[0].Request != "hot" || len(first.Admission.Requests) != 0 || !reflect.DeepEqual(first.PrefillDeferrals, []string{"resident"}) {
		t.Fatal(first)
	}
	if len(p.State().Waiting) != 0 {
		t.Fatal("proposal committed before feedback")
	}
	p.Observe(DecisionFeedback{Version: 1, Status: "superseded"})
	v.Version++
	decodeFillDecide(t, p, v)
	p.Observe(DecisionFeedback{Version: 2, Status: "applied", Promotions: []DecisionPromotionOutcome{{Request: "hot", StartedBlocks: 8}}})
	if len(p.State().Waiting) != 2 {
		t.Fatal("cohort missing", p.State())
	}
	v.Version++
	v.KV.Requests[2].TransferPending = true
	// Later arrivals cannot replace the already selected cold compute partner.
	v.Waiting = append(v.Waiting, DecisionRequest{ID: "new", State: StateQueued, InputTokens: 17, ArrivalUS: 2})
	v.KV.Requests = append(v.KV.Requests, DecisionKVRequest{ID: "new"})
	wait := decodeFillDecide(t, p, v)
	if len(wait.Promotions) != 0 || len(wait.Admission.Requests) != 0 || len(wait.PrefillDeferrals) != 1 {
		t.Fatal(wait)
	}
	p.Observe(DecisionFeedback{Version: 3, Status: "applied", Grants: []DecisionGrant{{Request: "decode", Tokens: 1}}})
	v.Version++
	v.KV.Requests[2].TransferPending = false
	v.KV.Requests[2].LocalPrefixBlocks = 8
	v.KV.Requests[2].LocalPrefixTokens = 128
	ready := decodeFillDecide(t, p, v)
	if len(ready.PrefillDeferrals) != 0 || !reflect.DeepEqual(ready.Admission.Requests, []string{"hot", "cold"}) || len(ready.QueueOrder) != 3 {
		t.Fatal(ready)
	}
	p.Observe(DecisionFeedback{Version: 4, Status: "applied", Grants: []DecisionGrant{{Request: "hot", Tokens: 1}, {Request: "cold", Tokens: 8}, {Request: "resident", Tokens: 8}}})
	if len(p.State().Waiting)+len(p.State().Resident) != 0 {
		t.Fatal("served cohort remained", p.State())
	}
}

func TestDecodeFillPartialRefusalAndDecodeLossDoNotCreateWaitLoop(t *testing.T) {
	for _, kind := range []string{"refused", "partial", "decode_ended", "cancelled"} {
		t.Run(kind, func(t *testing.T) {
			v := decodeFillView()
			p := decodeFillTestPolicy(t)
			decodeFillDecide(t, p, v)
			o := DecisionPromotionOutcome{Request: "hot", StartedBlocks: 8}
			if kind == "refused" {
				o.StartedBlocks, o.BlockedBlocks = 0, 8
			}
			if kind == "partial" {
				o.StartedBlocks, o.BlockedBlocks = 4, 4
			}
			p.Observe(DecisionFeedback{Version: 1, Status: "applied", Promotions: []DecisionPromotionOutcome{o}})
			v.Version++
			switch kind {
			case "partial":
				v.KV.Requests[2].LocalPrefixBlocks = 4
				v.KV.Requests[2].LocalPrefixTokens = 64
			case "decode_ended":
				v.Running = v.Running[1:]
				v.KV.Requests[2].TransferPending = true
			case "cancelled":
				v.Waiting = v.Waiting[1:]
			}
			plan := decodeFillDecide(t, p, v)
			if len(plan.PrefillDeferrals) != 0 || len(plan.Promotions) != 0 {
				t.Fatal("fallback still holds or reloads", plan)
			}
			p.Observe(DecisionFeedback{Version: 2, Status: "applied"})
			if kind == "refused" {
				v.Version++
				v.HBMFreeBlocks++
				if len(decodeFillDecide(t, p, v).Promotions) != 1 {
					t.Fatal("resource change did not allow retry")
				}
			}
		})
	}
}

func TestDecodeFillNoLoadAndComputeBalancedCasesUseOrdinaryPlan(t *testing.T) {
	for _, kind := range []string{"no_load", "balanced", "no_decode", "ineligible_resident"} {
		v := decodeFillView()
		p := decodeFillTestPolicy(t)
		switch kind {
		case "no_load":
			v.KV.Requests[2].RecoverablePrefixBlocks = 0
			v.KV.Requests[2].RecoverablePrefixTokens = 0
		case "balanced":
			p.base.MaxLoadComputeRatio = 1000
		case "no_decode":
			v.Running = v.Running[1:]
		case "ineligible_resident":
			v.Running[1].PrefillDeferrable = false
		}
		if got, want := decodeFillDecide(t, p, v), p.base.Decide(v); !reflect.DeepEqual(got, want) {
			t.Fatal(kind, got, want)
		}
	}
}
