package kv

import (
	"math"
	"reflect"
	"testing"
)

func futurePolicy(t *testing.T, mode string) *HotPrefixPolicy {
	t.Helper()
	zero := int64(0)
	p, err := NewHotPrefixPolicy(HotPrefixConfig{AgingIntervalRequests: 16, HBMEvictionUnit: "logical_segment", HBMScore: mode, ShadowTTLUS: &zero}, 16)
	if err != nil {
		t.Fatal(err)
	}
	p.future = &hotPrefixFuture{byHash: map[string][]*futurePlacementDemand{}, byRequest: map[string]map[string]*futurePlacementDemand{}, now: func() int64 { return 5 }}
	return p
}

func TestFutureNextUsesEarliestUnconsumedMemberAndLegalCandidates(t *testing.T) {
	p := futurePolicy(t, "oracle_next_arrival")
	c := logicalContext(p, []string{"a", "ab"}, []string{"b", "bc"})
	p.future.byHash["a"] = []*futurePlacementDemand{{at: 10}}
	p.future.byHash["ab"] = []*futurePlacementDemand{{at: 1000}}
	p.future.byHash["b"] = []*futurePlacementDemand{{at: 20}}
	d := p.Choose(c)
	if d.BlockID != 3 {
		t.Fatalf("must preserve soon-needed ancestor: %+v", d)
	}
	before := p.logicalSegments(c)
	p.future.byHash["a"][0].consumed = true
	if p.Choose(c).BlockID != 1 {
		t.Fatal("consumed reference retained as future demand")
	}
	after := p.logicalSegments(c)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("future scores changed segmentation")
	}
	c.Candidates = c.Candidates[:3] // protected bc makes the b chain illegal
	if p.Choose(c).BlockID != 1 {
		t.Fatal("oracle reclaimed protected ancestor")
	}
	if p.futureScore([]string{"never"}) != -math.MaxFloat64 {
		t.Fatal("missing no-use sentinel")
	}
}

func TestFutureRemainingCountsInputsOnceAndHandlesNonuniformSegments(t *testing.T) {
	p := futurePolicy(t, "oracle_remaining")
	c := logicalContext(p, []string{"a", "ab"}, []string{"b"})
	p.future.byHash["a"] = []*futurePlacementDemand{{request: "r1"}, {request: "r2"}}
	p.future.byHash["ab"] = []*futurePlacementDemand{{request: "r1"}, {request: "r2"}}
	p.future.byHash["b"] = []*futurePlacementDemand{{request: "r3"}}
	if p.Choose(c).BlockID != 2 {
		t.Fatal("did not normalize by physical segment size")
	}
	s := &PeerCache{hotprefix: &hotPrefixRuntime{policy: p}}
	p.future.byRequest["r1"] = map[string]*futurePlacementDemand{"a": p.future.byHash["a"][0], "ab": p.future.byHash["ab"][0]}
	s.consumeHotPrefixFuture("a", "r1")
	s.consumeHotPrefixFuture("a", "r1")
	s.consumeHotPrefixFuture("ab", "r1")
	if p.futureScore([]string{"a", "ab"}) != 1 || p.Choose(c).BlockID != 1 {
		t.Fatal("credit must be idempotent and ties must remain LRU")
	}
	s.consumeHotPrefixFuture("output-only", "r1") // future outputs are not indexed
}
