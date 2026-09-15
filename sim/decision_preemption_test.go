package sim

import (
	"math"
	"reflect"
	"testing"
)

func prefillFixture(t *testing.T, cost DecisionCostModel) (*Simulator, *[]DecisionRecord, *[]DecisionEventRecord) {
	t.Helper()
	s, records := serviceFixture(t, nil)
	s.maxNumSeqs, s.maxNumBatchedTokens, s.longPrefillTokenThreshold = 1, 2, 2
	if err := s.SetDecisionPolicy(&PrefillSJFPolicy{}, func(r DecisionRecord) { *records = append(*records, r) }); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableDecisionPrefillPreemption(); err != nil {
		t.Fatal(err)
	}
	var events []DecisionEventRecord
	if err := s.SetDecisionEventObserver(func(r DecisionEventRecord) { events = append(events, r) }); err != nil {
		t.Fatal(err)
	}
	if cost != nil {
		if err := s.SetDecisionCostModel(cost); err != nil {
			t.Fatal(err)
		}
	}
	return s, records, &events
}

func TestPrefillPreemptionAdmitsReplacementAndReentersOnce(t *testing.T) {
	for _, fee := range []int64{0, 7} {
		s, records, events := prefillFixture(t, LinearDecisionCost{PerPreemptionUS: fee, Provenance: "synthetic action sensitivity"})
		long, short := serviceRequest("long", 0), serviceRequest("short", 5)
		long.InputTokens = []TokenID{1, 2, 3, 4, 5, 6, 7, 8}
		short.InputTokens = []TokenID{21, 22}
		s.InjectArrival(long)
		s.InjectArrival(short)
		drainService(t, s)
		if long.State != StateCompleted || short.State != StateCompleted || s.Metrics.PreemptionCount != 1 {
			t.Fatal("requests did not complete with one preemption", long.State, short.State, s.Metrics.PreemptionCount)
		}
		if short.ArrivalTime+short.FirstTokenTime >= long.ArrivalTime+long.FirstTokenTime {
			t.Fatal("replacement did not finish first")
		}
		preemptions, first := 0, map[string]int{}
		for _, record := range *records {
			if len(record.Plan.Preemptions) == 0 {
				continue
			}
			preemptions++
			if record.Feedback.AtUS != 10+fee || record.Service.ExtraUS != fee || !reflect.DeepEqual(record.Feedback.Grants, []DecisionGrant{{Request: "short", Tokens: 2}}) {
				t.Fatal("replacement/service did not reach actual grant", record)
			}
			if !reflect.DeepEqual(record.Feedback.Preemptions, []DecisionPreemptionOutcome{{Request: "long", ComputedTokensBefore: 2, Status: "requeued"}}) {
				t.Fatal("missing actual preemption feedback", record.Feedback)
			}
		}
		observed := 0
		for _, record := range *events {
			if record.Event.Kind == "first_output" {
				first[record.Event.Request]++
			}
			if record.Event.Kind == "request_preempted" {
				observed++
				if record.Event.OccurredUS != 10+fee || record.Event.Request != "long" {
					t.Fatal("preemption timestamp/identity differs from application", record)
				}
			}
		}
		if preemptions != 1 || observed != 1 || first["long"] != 1 || first["short"] != 1 {
			t.Fatal("action/first output missing or duplicated", preemptions, observed, first)
		}
		if s.WaitQ.Len() != 0 || s.KVCache.UsedBlocks() != 0 {
			t.Fatal("reentry leaked queue membership or active KV references")
		}
	}
}

func TestPrefillPreemptionWithoutReplacementDoesNotStrandVictim(t *testing.T) {
	s, _, _ := prefillFixture(t, nil)
	issued := false
	s.decision.policy = &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
		p := DecisionPlan{Version: v.Version}
		if !issued && len(v.Running) > 0 && v.Running[0].PrefillPreemptible {
			p.Preemptions = []string{v.Running[0].ID}
			issued = true
		}
		return p
	}}
	r := serviceRequest("alone", 0)
	r.InputTokens = []TokenID{1, 2, 3, 4, 5, 6}
	s.InjectArrival(r)
	drainService(t, s)
	if !issued || r.State != StateCompleted || s.Metrics.PreemptionCount != 1 {
		t.Fatal("all-preempted empty batch lost its wakeup", issued, r.State)
	}
}

func TestPrefillPreemptionPlanAndFeedbackAreDetached(t *testing.T) {
	p := DecisionPlan{Preemptions: []string{"a"}}
	c := cloneDecisionPlan(p)
	c.Preemptions[0] = "b"
	f := DecisionFeedback{Preemptions: []DecisionPreemptionOutcome{{Request: "a", Status: "requeued"}}}
	g := cloneDecisionFeedback(f)
	g.Preemptions[0].Request = "b"
	if p.Preemptions[0] != "a" || f.Preemptions[0].Request != "a" {
		t.Fatal("strategy can rewrite runtime action evidence")
	}
}

func TestPrefillPreemptionMultipleVictimsPreserveReentryOrder(t *testing.T) {
	s, records, _ := prefillFixture(t, nil)
	s.maxNumSeqs = 2
	for index, id := range []string{"a", "b"} {
		r := serviceRequest(id, 0)
		r.InputTokens = []TokenID{TokenID(index + 1), 2, 3, 4, 5, 6}
		r.State = StateRunning
		if !s.KVCache.AllocateKVBlocks(r, 0, 2, nil) {
			t.Fatal("allocation failed")
		}
		r.ProgressIndex = 2
		s.reqNumComputedTokens[id] = 2
		s.RunningBatch.Requests = append(s.RunningBatch.Requests, r)
	}
	s.WaitQ.Enqueue(serviceRequest("short", 0))
	s.decision.policy = &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
		return DecisionPlan{Version: v.Version, Preemptions: []string{"b", "a"}, QueueOrder: []string{"short"}, Admission: &DecisionAdmission{Requests: []string{"short"}}, BatchTokenCap: 1}
	}}
	caps := s.prepareDecision(10)
	s.formScheduledBatch(10, caps, nil)
	if len(*records) != 1 || len((*records)[0].Feedback.Preemptions) != 2 || s.Metrics.PreemptionCount != 2 {
		t.Fatal("multi-victim action missing")
	}
	queue := s.WaitQ.Items()
	if len(queue) != 2 || queue[0].ID != "b" || queue[1].ID != "a" {
		t.Fatal("reentry order differs from plan", queue)
	}
	if !reflect.DeepEqual((*records)[0].Feedback.Grants, []DecisionGrant{{Request: "short", Tokens: 1}}) {
		t.Fatal("yielded requests consumed aggregate budget")
	}
}

func TestPrefillPreemptionKeepsLateArrivalsOutsidePaidPlan(t *testing.T) {
	s, records, _ := prefillFixture(t, LinearDecisionCost{PerPreemptionUS: 20, Provenance: "synthetic"})
	long, short, late := serviceRequest("long", 0), serviceRequest("short", 5), serviceRequest("late", 15)
	long.InputTokens = []TokenID{1, 2, 3, 4, 5, 6, 7, 8}
	short.InputTokens = []TokenID{21, 22}
	late.InputTokens = []TokenID{31}
	s.InjectArrival(long)
	s.InjectArrival(short)
	s.InjectArrival(late)
	drainService(t, s)
	for _, r := range *records {
		if len(r.Plan.Preemptions) > 0 && r.View.NowUS == 10 {
			if r.Feedback.AtUS != 30 || !reflect.DeepEqual(r.Feedback.Grants, []DecisionGrant{{Request: "short", Tokens: 2}}) {
				t.Fatal("late arrival entered the paid plan", r)
			}
		}
	}
	if long.State != StateCompleted || short.State != StateCompleted || late.State != StateCompleted {
		t.Fatal("late arrival or reentry stranded")
	}
}

func TestPrefillPreemptionRejectsEntirePlanBeforeMutation(t *testing.T) {
	for _, mode := range []string{"unknown", "repeat", "decode", "already_output", "cap", "queue", "admission", "disabled"} {
		t.Run(mode, func(t *testing.T) {
			s, _, _ := prefillFixture(t, nil)
			r := serviceRequest("running", 0)
			r.InputTokens = []TokenID{1, 2, 3, 4, 5, 6}
			r.State = StateRunning
			if !s.KVCache.AllocateKVBlocks(r, 0, 2, nil) {
				t.Fatal("allocation failed")
			}
			r.ProgressIndex = 2
			s.reqNumComputedTokens[r.ID] = 2
			s.RunningBatch = &Batch{Requests: []*Request{r}}
			s.WaitQ.Enqueue(serviceRequest("waiting", 0))
			if mode == "decode" {
				r.ProgressIndex = 6
				r.TTFTSet = true
			}
			if mode == "already_output" {
				s.decision.events.firstOutput[r.ID] = true
			}
			if mode == "disabled" {
				s.decision.prefillPreemption = false
			}
			before := s.decisionView(10)
			used := s.KVCache.UsedBlocks()
			s.decision.policy = &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
				p := DecisionPlan{Version: v.Version, Preemptions: []string{"running"}}
				switch mode {
				case "unknown":
					p.Preemptions = []string{"future"}
				case "repeat":
					p.Preemptions = append(p.Preemptions, "running")
				case "cap":
					p.TokenCaps = []DecisionTokenCap{{Request: "running", Tokens: 1}}
				case "queue":
					p.QueueOrder = []string{"future"}
				case "admission":
					p.Admission = &DecisionAdmission{Requests: []string{"running"}}
				}
				return p
			}}
			func() {
				defer func() {
					if recover() == nil {
						t.Error("invalid plan accepted")
					}
				}()
				s.prepareDecision(10)
			}()
			if !reflect.DeepEqual(before, s.decisionView(10)) || s.KVCache.UsedBlocks() != used || s.Metrics.PreemptionCount != 0 {
				t.Fatal("rejected plan changed queue/progress/resources")
			}
		})
	}
}

func TestPrefillPreemptionCostCoverageAndOverflow(t *testing.T) {
	v := DecisionView{MaxBatchTokens: 8, Capabilities: DecisionCapabilities{PrefillPreemption: true}, KV: DecisionKVState{TransfersKnown: true}, Running: []DecisionRequest{{ID: "a", State: StateRunning, ComputedTokens: 2, InputTokens: 8, PrefillPreemptible: true}}}
	p := DecisionPlan{Preemptions: []string{"a"}}
	c := LinearDecisionCost{FixedUS: 3, PerPreemptionUS: 7, Provenance: "synthetic", Limits: &DecisionCostLimits{MaxVisibleRequests: 2, MaxInputTokens: 8}}
	if _, err := c.Estimate(v, p); err == nil {
		t.Fatal("old profile accepted new capability")
	}
	c.Limits.RequirePrefillPreemption = true
	warned := false
	c.SetProfileWarningObserver(func(parameter string, observed, maximum int64) {
		warned = parameter == "preemptions" && observed == 1 && maximum == 0
	})
	if cost, err := c.Estimate(v, p); err != nil || cost.ExtraUS != 10 || !warned {
		t.Fatal("preemption extrapolation rejected, silent or uncharged", cost, err)
	}
	c.Limits.MaxPreemptions = 1
	if got, err := c.Estimate(v, p); err != nil || got.ExtraUS != 10 {
		t.Fatal("action cost missing", got, err)
	}
	c.FixedUS = math.MaxInt64
	if _, err := c.Estimate(v, p); err == nil {
		t.Fatal("overflow accepted")
	}
}

func TestPrefillPreemptionReleasesSequenceBeforeAggregateCap(t *testing.T) {
	v := DecisionView{MaxBatchTokens: 8, Capabilities: DecisionCapabilities{PrefillPreemption: true, BatchTokenCap: true}, Running: []DecisionRequest{{ID: "a", State: StateRunning, ComputedTokens: 2, InputTokens: 8, PrefillPreemptible: true}, {ID: "b", State: StateRunning, ComputedTokens: 2, InputTokens: 8, PrefillPreemptible: true}}}
	if err := ValidateDecision(v, DecisionPlan{Preemptions: []string{"a"}, BatchTokenCap: 1}); err != nil {
		t.Fatal("cap counts a yielded request", err)
	}
	if ValidateDecision(v, DecisionPlan{BatchTokenCap: 1}) == nil {
		t.Fatal("ordinary running request starved")
	}
}

func TestPrefillPreemptionTimeoutDuringServiceSupersedesAction(t *testing.T) {
	s, records, events := prefillFixture(t, LinearDecisionCost{PerPreemptionUS: 20, Provenance: "synthetic"})
	long, short := serviceRequest("long", 0), serviceRequest("short", 5)
	long.InputTokens = []TokenID{1, 2, 3, 4, 5, 6, 7, 8}
	long.Deadline = 15
	short.InputTokens = []TokenID{21, 22}
	s.InjectArrival(long)
	s.InjectArrival(short)
	drainService(t, s)
	if long.State != StateTimedOut || short.State != StateCompleted || s.Metrics.PreemptionCount != 0 {
		t.Fatal("expired victim was preempted", long.State, short.State)
	}
	found := false
	for _, r := range *records {
		if len(r.Plan.Preemptions) > 0 {
			found = true
			if r.Feedback.Status != "superseded" || r.Feedback.AtUS != 30 || len(r.Feedback.Preemptions) > 0 {
				t.Fatal("expired plan changed resources/refunded service", r)
			}
		}
	}
	for _, e := range *events {
		if e.Event.Kind == "request_preempted" || e.Event.Kind == "first_output" && e.Event.Request == "long" {
			t.Fatal("phantom action/output", e)
		}
	}
	if !found {
		t.Fatal("test missed in-service timeout")
	}
}

func TestPrefillSJFDoesNotReadFutureOrPreemptPendingWork(t *testing.T) {
	v := DecisionView{MaxSequences: 1, Capabilities: DecisionCapabilities{PrefillPreemption: true}, Waiting: []DecisionRequest{{ID: "short", InputTokens: 2}}, Running: []DecisionRequest{{ID: "long", InputTokens: 8, ComputedTokens: 2, State: StateRunning, PrefillPreemptible: true}}}
	p := &PrefillSJFPolicy{}
	if !reflect.DeepEqual(p.Decide(v).Preemptions, []string{"long"}) {
		t.Fatal("missed shorter request")
	}
	v.KV.Requests = []DecisionKVRequest{{ID: "short", TransferPending: true}}
	if len(p.Decide(v).Preemptions) > 0 {
		t.Fatal("yielded to a blocked request")
	}
	v.KV.Requests = nil
	v.Running[0].PrefillPreemptible = false
	if len(p.Decide(v).Preemptions) > 0 {
		t.Fatal("policy preempted unsupported running state")
	}
}
