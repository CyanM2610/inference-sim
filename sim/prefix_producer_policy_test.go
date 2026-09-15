package sim

import (
	"reflect"
	"testing"
)

func producerView() DecisionView {
	v := DecisionView{NowUS: 10, BlockTokens: 16, MaxBatchTokens: 128, MaxSequences: 4, PrefillChunk: 32,
		Capabilities: DecisionCapabilities{QueueOrder: true, PrefixProducers: true, AdmissionSelection: true, TokenCaps: true, RestoreChoice: true},
		KV:           DecisionKVState{PrefixStateKnown: true}}
	for _, id := range []string{"a", "b", "c"} {
		v.Waiting = append(v.Waiting, DecisionRequest{ID: id, State: StateQueued, InputTokens: 64})
		v.KV.Requests = append(v.KV.Requests, DecisionKVRequest{ID: id})
		for _, p := range []string{"a", "b", "c"} {
			if p != id {
				v.PrefixProducers = append(v.PrefixProducers, DecisionPrefixProducer{id, p, 3})
			}
		}
	}
	return v
}

func TestProducerWaitBreaksCyclesAndReleasesOnFactsOrAge(t *testing.T) {
	base, _ := NewBalancedBatchPolicy(100, 128, 4, 0)
	p, _ := NewProducerWaitPolicy(base, 32, 100)
	v := producerView()
	plan := p.Decide(v)
	if err := ValidateDecision(v, plan); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.Admission.Requests, []string{"a"}) {
		t.Fatal("waiting dependencies cycle or root is held", plan)
	}
	// A producer finishes/cancels: it disappears, so the next root is admitted.
	v.Waiting = v.Waiting[1:]
	v.PrefixProducers = []DecisionPrefixProducer{{"b", "c", 3}, {"c", "b", 3}}
	if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"b"}) {
		t.Fatal("missing producer retained", got)
	}
	// Ready slow-tier copies already cover the shared prefix: use normal restore.
	v.KV.Requests[2].RecoverablePrefixBlocks = 3
	v.KV.Requests[2].RecoverablePrefixTokens = 48
	if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"b", "c"}) {
		t.Fatal("ready copy was ignored", got)
	}
	v = producerView()
	v.NowUS = 100
	if len(p.Decide(v).Admission.Requests) != 3 {
		t.Fatal("arrival-relative timeout did not release")
	}
}

func TestProducerWaitProtectsExistingConsumerAndHonorsSharedThreshold(t *testing.T) {
	base, _ := NewBalancedBatchPolicy(100, 128, 4, 0)
	p, _ := NewProducerWaitPolicy(base, 32, 100)
	for _, change := range []func(*DecisionView){
		func(v *DecisionView) { v.Waiting[1].ComputedTokens = 16 },
		func(v *DecisionView) { v.KV.Requests[1].LocalPrefixBlocks = 3; v.KV.Requests[1].LocalPrefixTokens = 48 },
		func(v *DecisionView) { v.PrefixProducers = []DecisionPrefixProducer{{"b", "a", 1}} },
	} {
		v := producerView()
		v.Waiting = v.Waiting[:2]
		change(&v)
		if got := p.Decide(v).Admission.Requests; !reflect.DeepEqual(got, []string{"a", "b"}) {
			t.Fatal("ineligible consumer held", got)
		}
	}
	// A later-arriving running producer can be awaited; it cannot join a wait cycle.
	v := producerView()
	v.Running = []DecisionRequest{v.Waiting[2]}
	v.Running[0].ArrivalUS = 5
	v.Waiting = v.Waiting[:2]
	if len(p.Decide(v).Admission.Requests) != 0 {
		t.Fatal("running producer was ignored")
	}
}

func TestProducerSnapshotUsesOnlyArrivedInputsAndDetachedIdentity(t *testing.T) {
	s := decisionFixture()
	s.decision = &decisionRuntime{prefixProducers: true}
	s.WaitQ.Items()[1].InputTokens = []TokenID{1, 2, 3, 9}
	a := s.decisionView(10)
	if !reflect.DeepEqual(a.PrefixProducers, []DecisionPrefixProducer{{"a", "b", 1}, {"b", "a", 1}}) {
		t.Fatal(a.PrefixProducers)
	}
	s.WaitQ.Peek().OutputTokens = make([]TokenID, 9999)
	s.Schedule(&ArrivalEvent{time: 1000, Request: &Request{ID: "future", InputTokens: []TokenID{1, 2, 3, 4, 5}}})
	if !reflect.DeepEqual(a, s.decisionView(10)) {
		t.Fatal("future inputs/outputs affected producer state")
	}
	c := cloneDecisionView(a)
	c.PrefixProducers[0].Producer = "mutated"
	if a.PrefixProducers[0].Producer != "b" {
		t.Fatal("producer slice aliases validation snapshot")
	}
	queries := []DecisionKVQuery{{ID: "a", Input: []TokenID{1, 2, 3, 4}}, {ID: "b", Input: []TokenID{9, 2, 3, 4}}, {ID: "c", Input: []TokenID{1, 2, 8, 4}}}
	got := decisionPrefixProducers([]DecisionRequest{{ID: "a"}}, queries, 2)
	if !reflect.DeepEqual(got, []DecisionPrefixProducer{{"a", "c", 1}}) {
		t.Fatal("non-prefix token match used", got)
	}
}
