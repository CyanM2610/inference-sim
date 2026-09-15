package sim

import (
	"reflect"
	"testing"
)

func budgetQueueView() DecisionView {
	v := DecisionView{Version: 1, BlockTokens: 16, MaxBatchTokens: 7, MaxSequences: 2, PrefillChunk: 8,
		Capabilities: DecisionCapabilities{RestoreChoice: true, CostEstimates: true, QueueOrder: true,
			TokenCaps: true, AdmissionSelection: true, PrefillPreemption: true, CapacityPreemption: true},
		KV:        DecisionKVState{PrefixStateKnown: true, TransfersKnown: true},
		Estimates: &DecisionEstimates{Provenance: "synthetic queue test", Coverage: "uncalibrated"}}
	for _, r := range []DecisionRequest{{ID: "primary", InputTokens: 10000, TTFTTargetUS: 1000000000},
		{ID: "best", InputTokens: 10000, TTFTTargetUS: 1}} {
		r.State, r.ClientOutputLimit = StateQueued, 4
		v.Waiting = append(v.Waiting, r)
		v.KV.Requests = append(v.KV.Requests, DecisionKVRequest{ID: r.ID})
		v.Estimates.Requests = append(v.Estimates.Requests, DecisionRequestEstimate{Request: r.ID,
			Choices: []DecisionRestoreEstimate{{ComputeUS: 1000, MaxLoadComputeUS: 1000}}})
	}
	return v
}

func capsByRequest(p DecisionPlan) map[string]int64 {
	caps := map[string]int64{}
	for _, cap := range p.TokenCaps {
		caps[cap.Request] = cap.Tokens
	}
	return caps
}

func TestBudgetQueuesRepayRoundingUsingActualPrefillGrants(t *testing.T) {
	p, err := NewBudgetQueuePolicy(1, 0, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	v := budgetQueueView()
	for step := 0; step < 200; step++ {
		plan := p.Decide(v)
		if err := ValidateDecision(v, plan); err != nil {
			t.Fatal(err)
		}
		caps := capsByRequest(plan)
		if caps["primary"] < 1 || caps["best"] < 1 || caps["primary"]+caps["best"] != 7 {
			t.Fatal("lost runnable work or queue service", caps)
		}
		if step == 0 && p.ServiceState().PrimaryTokens != 0 {
			t.Fatal("proposal counted as service")
		}
		p.Observe(DecisionFeedback{Version: v.Version, Status: "applied", Grants: []DecisionGrant{
			{Request: "primary", Tokens: caps["primary"]}, {Request: "best", Tokens: caps["best"]}}})
		v.Running = append(v.Running, v.Waiting...)
		v.Waiting = nil
		v.Estimates.Requests = nil
		for i := range v.Running {
			v.Running[i].State = StateRunning
			v.Running[i].ComputedTokens += caps[v.Running[i].ID]
			v.Running[i].PrefillPreemptible = true
		}
		v.Version++
		v.NowUS++
	}
	s := p.ServiceState()
	if s.BestEffortTokens != 350 || s.PrimaryTokens != 1050 || s.DeficitUnits != 0 || s.ContendedBestTokens != 350 {
		t.Fatal("actual fractional service drifted", s)
	}
}

func TestBudgetQueuesFailedServiceIncreasesNextTargetWithoutInventingGrants(t *testing.T) {
	p, _ := NewBudgetQueuePolicy(1, 0, 3, 1)
	v := budgetQueueView()
	first := p.Decide(v)
	p.Observe(DecisionFeedback{Version: 1, Status: "applied", Grants: []DecisionGrant{{Request: "primary", Tokens: 5}},
		Unselected: []DecisionUnselected{{Request: "best", Reason: "runtime_not_selected"}}})
	s := p.ServiceState()
	if s.BestEffortTokens != 0 || s.DeficitUnits != 5 || s.BestEffortUnservedRounds != 1 {
		t.Fatal(s)
	}
	v.Version++
	v.NowUS++
	second := p.Decide(v)
	if capsByRequest(second)["best"] <= capsByRequest(first)["best"] {
		t.Fatal("unserved work did not affect next decision")
	}
	// No compute grants: restore/wait/control work cannot repay service debt.
	p.Observe(DecisionFeedback{Version: v.Version, Status: "applied"})
	if p.ServiceState().DeficitUnits != 5 || p.ServiceState().BestEffortTokens != 0 {
		t.Fatal("empty control turn created service")
	}
}

func TestBudgetQueuesProtectDecodeAndExcludeItFromPrefillShare(t *testing.T) {
	p, _ := NewBudgetQueuePolicy(1, 0, 3, 1)
	v := budgetQueueView()
	v.MaxSequences = 3
	v.Running = []DecisionRequest{{ID: "decode", State: StateRunning, InputTokens: 8, ComputedTokens: 8, EmittedTokens: 1, ClientOutputLimit: 4}}
	v.KV.Requests = append(v.KV.Requests, DecisionKVRequest{ID: "decode"})
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	caps := capsByRequest(plan)
	if caps["decode"] != 1 || caps["primary"]+caps["best"] != 6 {
		t.Fatal(caps)
	}
	for _, id := range plan.CapacityVictims.Order {
		if id == "decode" {
			t.Fatal("decode victim")
		}
	}
	var grants []DecisionGrant
	for id, tokens := range caps {
		grants = append(grants, DecisionGrant{Request: id, Tokens: tokens})
	}
	p.Observe(DecisionFeedback{Version: v.Version, Status: "applied", Grants: grants})
	s := p.ServiceState()
	if s.PrimaryTokens+s.BestEffortTokens != 6 {
		t.Fatal("decode counted as prefill service", s)
	}
}

func TestBudgetQueuesSlotBlockingIsReportedNotPromisedAway(t *testing.T) {
	p, _ := NewBudgetQueuePolicy(1, 0, 3, 1)
	v := budgetQueueView()
	v.MaxSequences = 1
	initial := v.Waiting[0]
	v.Waiting = v.Waiting[:1]
	v.KV.Requests = v.KV.Requests[:1]
	v.Estimates.Requests = v.Estimates.Requests[:1]
	plan := p.Decide(v)
	p.Observe(DecisionFeedback{Version: 1, Status: "applied", Grants: []DecisionGrant{{Request: "primary", Tokens: 7}}})
	v = budgetQueueView()
	v.Version = 2
	v.NowUS = 1
	v.MaxSequences = 1
	initial.State, initial.ComputedTokens, initial.PrefillPreemptible = StateRunning, 7, true
	v.Running = []DecisionRequest{initial}
	v.Waiting = v.Waiting[1:]
	v.Estimates.Requests = v.Estimates.Requests[1:]
	plan = p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if len(plan.Admission.Requests) != 0 || len(plan.Preemptions) != 0 {
		t.Fatal("fabricated a sequence slot", plan)
	}
	p.Observe(DecisionFeedback{Version: 2, Status: "applied", Grants: []DecisionGrant{{Request: "primary", Tokens: 7}}})
	if p.ServiceState().BestEffortUnservedRounds != 1 || p.ServiceState().DeficitUnits != 7 {
		t.Fatal(p.ServiceState())
	}
}

func TestBudgetQueuesFeedbackValidationAndNoIdleCredit(t *testing.T) {
	p, _ := NewBudgetQueuePolicy(1, 0, 3, 1)
	v := budgetQueueView()
	p.Decide(v)
	before := p.ServiceState()
	for _, f := range []DecisionFeedback{
		{Version: 2, Status: "applied"}, {Version: 1, Status: "unknown"},
		{Version: 1, Status: "applied", Grants: []DecisionGrant{{Request: "absent", Tokens: 1}}},
		{Version: 1, Status: "applied", Grants: []DecisionGrant{{Request: "primary", Tokens: 8}}},
		{Version: 1, Status: "superseded", Grants: []DecisionGrant{{Request: "primary", Tokens: 1}}},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("accepted invalid feedback", f)
				}
			}()
			p.Observe(f)
		}()
		if !reflect.DeepEqual(before, p.ServiceState()) {
			t.Fatal("invalid feedback partially changed state")
		}
	}
	p.Observe(DecisionFeedback{Version: 1, Status: "superseded"})
	if p.ServiceState() != before {
		t.Fatal("superseded plan was charged")
	}
	v.Version++
	v.Waiting = v.Waiting[1:]
	v.KV.Requests = v.KV.Requests[1:]
	v.Estimates.Requests = v.Estimates.Requests[1:]
	plan := p.Decide(v)
	if capsByRequest(plan)["best"] != 7 {
		t.Fatal("unused primary share was not borrowed")
	}
	p.Observe(DecisionFeedback{Version: 2, Status: "applied", Grants: []DecisionGrant{{Request: "best", Tokens: 7}}})
	if p.ServiceState().DeficitUnits != 0 {
		t.Fatal("noncontended service generated credit")
	}
	p.Reset()
	if p.ServiceState() != (BudgetQueueService{WeightSum: 4}) {
		t.Fatal("reset retained service history")
	}
}
