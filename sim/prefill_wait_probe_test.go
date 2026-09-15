package sim

import (
	"math"
	"reflect"
	"strconv"
	"testing"
)

type prefillWaitProbeBase struct {
	plan      DecisionPlan
	feedback  []DecisionFeedback
	resets    int
	decisions int
}

func (p *prefillWaitProbeBase) Decide(DecisionView) DecisionPlan {
	p.decisions++
	return p.plan
}
func (p *prefillWaitProbeBase) Observe(f DecisionFeedback) { p.feedback = append(p.feedback, f) }
func (p *prefillWaitProbeBase) Reset()                     { p.resets++ }

func prefillWaitProbeView() DecisionView {
	return DecisionView{Version: 3, NowUS: 40, Capabilities: DecisionCapabilities{PrefillWaits: true},
		Waiting: []DecisionRequest{{ID: "queued", PrefillWaitable: true}},
		Running: []DecisionRequest{{ID: "a", PrefillWaitable: true}, {ID: "b", PrefillWaitable: true},
			{ID: "preempted", PrefillWaitable: true}, {ID: "decode"}},
	}
}

func TestPrefillWaitProbeSelectionPreservesBasePlan(t *testing.T) {
	for _, tc := range []struct {
		name      string
		selection []string
		want      []string
	}{
		{"all", nil, []string{"a", "b"}},
		{"empty", []string{}, nil},
		{"subset", []string{"b", "b", "queued", "decode", "unknown"}, []string{"b"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			storage := []DecisionWait{{Request: "queued", UntilUS: 80}, {Request: "sentinel", UntilUS: 99}}
			at := int64(90)
			base := &prefillWaitProbeBase{plan: DecisionPlan{Version: 3, Waits: storage[:1],
				Preemptions: []string{"preempted"}, QueueOrder: []string{"queued"}, RevisitAtUS: &at,
				Admission: &DecisionAdmission{Requests: []string{}}, TokenCaps: []DecisionTokenCap{{Request: "a", Tokens: 8}},
				Restores: []DecisionRestore{{Request: "queued", MaxPrefixBlocks: 0}}, BatchTokenCap: 16}}
			probe, err := NewPrefillWaitProbe(base, 10, tc.selection)
			if err != nil {
				t.Fatal(err)
			}
			v := prefillWaitProbeView()
			want := cloneDecisionPlan(base.plan)
			for _, id := range tc.want {
				want.Waits = append(want.Waits, DecisionWait{Request: id, UntilUS: 50})
			}
			if got := probe.Decide(v); !reflect.DeepEqual(got, want) {
				t.Fatalf("plan = %+v, want %+v", got, want)
			}
			if len(base.plan.Waits) != 1 || storage[1].Request != "sentinel" || base.decisions != 1 {
				t.Fatal("probe modified base plan storage or bypassed base decision", base)
			}
			if got := probe.Decide(v); !reflect.DeepEqual(got, base.plan) {
				t.Fatal("probe issued a second wait", got)
			}
		})
	}
}

func TestPrefillWaitProbePreemptionResetAndFeedback(t *testing.T) {
	base := &prefillWaitProbeBase{plan: DecisionPlan{Version: 3, Preemptions: []string{"a"}}}
	selection := []string{"a"}
	probe, err := NewPrefillWaitProbe(base, 10, selection)
	if err != nil {
		t.Fatal(err)
	}
	selection[0] = "b" // The configured request set is owned by the probe.
	v := prefillWaitProbeView()
	if len(probe.Decide(v).Waits) != 0 || base.resets != 1 {
		t.Fatal("preemption did not take precedence or construction did not reset base")
	}
	base.plan.Preemptions = nil
	if got := probe.Decide(v).Waits; !reflect.DeepEqual(got, []DecisionWait{{Request: "a", UntilUS: 50}}) {
		t.Fatal("skipped preemption incorrectly consumed one-shot wait", got)
	}
	f := DecisionFeedback{Version: 3, Status: "rejected", Reason: "fixture"}
	probe.Observe(f)
	v.NowUS = 100
	if len(probe.Decide(v).Waits) != 0 || !reflect.DeepEqual(base.feedback, []DecisionFeedback{f}) {
		t.Fatal("feedback was lost or reset the one-shot issuance")
	}
	probe.Reset()
	if got := probe.Decide(v).Waits; !reflect.DeepEqual(got, []DecisionWait{{Request: "a", UntilUS: 110}}) || base.resets != 2 {
		t.Fatal("reset did not clear probe and forward to base", got, base.resets)
	}
}

func TestPrefillWaitProbeZeroDurationAndInvalidInputs(t *testing.T) {
	base := &prefillWaitProbeBase{plan: DecisionPlan{Version: 3, Waits: []DecisionWait{}, Preemptions: []string{},
		QueueOrder: []string{}, Admission: &DecisionAdmission{Requests: []string{}}}}
	probe, err := NewPrefillWaitProbe(base, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if got := probe.Decide(prefillWaitProbeView()); !reflect.DeepEqual(got, base.plan) {
			t.Fatal("zero duration changed base plan", got)
		}
	}
	if _, err := NewPrefillWaitProbe(nil, 0, nil); err == nil {
		t.Fatal("nil base accepted")
	}
	if _, err := NewPrefillWaitProbe(base, -1, nil); err == nil {
		t.Fatal("negative duration accepted")
	}
	for _, duration := range []int64{0, 10} {
		t.Run(strconv.FormatInt(duration, 10), func(t *testing.T) {
			probe, _ := NewPrefillWaitProbe(base, duration, nil)
			before := base.decisions
			defer func() {
				if recover() == nil || base.decisions != before {
					t.Fatal("missing capability must fail before calling the base")
				}
			}()
			probe.Decide(DecisionView{})
		})
	}
}

func TestPrefillWaitProbeRejectsDeadlineOverflow(t *testing.T) {
	base, _ := NewQueueDecisionPolicy("fcfs", 0)
	probe, _ := NewPrefillWaitProbe(base, 10, nil)
	v := prefillWaitProbeView()
	v.NowUS = math.MaxInt64 - 5
	defer func() {
		if recover() == nil {
			t.Fatal("overflowing deadline accepted")
		}
	}()
	probe.Decide(v)
}

func TestPrefillWaitProbeWaitsForObservedProgress(t *testing.T) {
	base := &prefillWaitProbeBase{}
	probe, err := NewPrefillWaitProbeWithOptions(base, 100, []string{"a"}, PrefillWaitProbeOptions{MinComputedTokens: 32})
	if err != nil {
		t.Fatal(err)
	}
	v := prefillWaitProbeView()
	v.Running[0].ComputedTokens = 16
	if got := probe.Decide(v); len(got.Waits) != 0 {
		t.Fatal("wait issued before observed threshold", got)
	}
	v.NowUS = 80
	v.Running[0].ComputedTokens = 32
	got := probe.Decide(v)
	if !reflect.DeepEqual(got.Waits, []DecisionWait{{Request: "a", UntilUS: 180}}) {
		t.Fatal("threshold did not issue wait", got)
	}
	v.Running[0].ComputedTokens = 48
	if got := probe.Decide(v); len(got.Waits) != 0 {
		t.Fatal("threshold wait repeated", got)
	}
	probe.Reset()
	v.Running[0].ComputedTokens = 16
	if got := probe.Decide(v); len(got.Waits) != 0 {
		t.Fatal("reset lost threshold", got)
	}
	if _, err := NewPrefillWaitProbeWithOptions(base, 1, nil, PrefillWaitProbeOptions{MinComputedTokens: -1}); err == nil {
		t.Fatal("negative threshold accepted")
	}
}

func TestPrefillWaitProbeResetStartsNewBudgetWorkload(t *testing.T) {
	constructors := map[string]func(float64, int64) (*BudgetRestorePolicy, error){
		"initial_budget": NewInitialBudgetRestorePolicy, "budget_pressure": NewBudgetPressurePolicy,
		"budget_pressure_fcfs": NewBudgetPressureFIFOPolicy, "restore_bound": NewBudgetRestorePolicy,
		"completion_bound": NewCompletionBudgetRestorePolicy,
	}
	for name, construct := range constructors {
		t.Run(name, func(t *testing.T) {
			base, _ := construct(1, 2)
			retained := base.initialBudgets != nil
			probe, _ := NewPrefillWaitProbe(base, 10, nil)
			view := initialBudgetView()
			view.Capabilities.PrefillWaits, view.Capabilities.WaitUntil = true, true
			view.Capabilities.PrefillPreemption, view.Capabilities.CapacityPreemption = true, true
			probe.Decide(view)
			request := view.Waiting[0]
			request.State, request.ComputedTokens = StateRunning, 2
			request.PrefillWaitable, request.PrefillPreemptible = true, true
			view.Waiting, view.Running, view.Estimates.Requests = nil, []DecisionRequest{request}, nil
			view.NowUS = 20
			if len(probe.Decide(view).Waits) != 1 {
				t.Fatal("first workload did not issue wait")
			}
			probe.Reset()
			if budgets := base.RequestBudgets(0); len(budgets) != 0 || (base.initialBudgets != nil) != retained {
				t.Fatal("reset retained a request/clock or changed policy mode", budgets)
			}
			// The same ID is valid in a new run with a different identity, earlier
			// clock and intrinsic estimate. Compare with a separately fresh policy.
			view = initialBudgetView()
			view.Capabilities.PrefillWaits, view.Capabilities.WaitUntil = true, true
			view.Capabilities.PrefillPreemption, view.Capabilities.CapacityPreemption = true, true
			view.NowUS, view.Waiting[0].ArrivalUS = 5, 1
			view.Waiting[0].InputTokens, view.Waiting[0].TTFTTargetUS = 6, 80
			choices := view.Estimates.Requests[0].Choices
			choices[len(choices)-1].ComputeUS, choices[len(choices)-1].MaxLoadComputeUS = 4, 6
			fresh, _ := construct(1, 2)
			if got, want := probe.Decide(view), fresh.Decide(view); !reflect.DeepEqual(got, want) {
				t.Fatal("reset did not preserve configured policy behavior", got, want)
			}
			if got, want := base.RequestBudgets(5), fresh.RequestBudgets(5); !reflect.DeepEqual(got, want) || retained && got[0].InitialUS != 75 {
				t.Fatal("new workload did not reinitialize budget", got, want)
			}
			request = view.Waiting[0]
			request.State, request.ComputedTokens, request.PrefillWaitable = StateRunning, 2, true
			view.Waiting, view.Running, view.Estimates.Requests = nil, []DecisionRequest{request}, nil
			view.NowUS = 6
			if got := probe.Decide(view).Waits; len(got) != 1 || got[0].UntilUS != 16 {
				t.Fatal("reset did not reissue wait in new workload", got)
			}
		})
	}
}
