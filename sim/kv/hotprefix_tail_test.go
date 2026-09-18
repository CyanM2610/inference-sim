package kv

import (
	"reflect"
	"testing"
)

func tailPolicy(t *testing.T) *HotPrefixPolicy {
	t.Helper()
	p, err := NewHotPrefixPolicy(HotPrefixConfig{AgingIntervalRequests: 16, AdmissionThreshold: 2,
		HBMEvictionUnit: "logical_segment", HBMScore: "benefit_next", ReclaimMode: "deficit_tail",
		Benefit: &HotPrefixBenefitConfig{HorizonUS: 1000, DecayUS: 1000}, Diagnostics: &HotPrefixDiagnosticsConfig{TraceCandidates: true}}, 16)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTailChoiceIdentifiesLengthAndPreservesRemainingPrefix(t *testing.T) {
	p := tailPolicy(t)
	c := logicalContext(p, []string{"a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdefg", "abcdefgh"})
	c.DeficitBlocks = 3
	c.BenefitCosts = benefitState(t, c)
	d := p.Choose(c)
	var ids []int64
	for _, a := range d.Reclaim {
		ids = append(ids, a.BlockID)
	}
	if !reflect.DeepEqual(ids, []int64{7, 6, 5}) {
		t.Fatal("did not choose the exact legal tail", ids)
	}
	r := p.diagnostics.Records[0]
	if !reflect.DeepEqual(r.SelectedMemberBlocks, ids) || r.CandidateUnit != "logical_tail" || len(r.Candidates) != 3 {
		t.Fatal("tail alternatives lost identity", r)
	}
	for _, x := range r.Candidates {
		if len(x.ResidentMemberBlocks) != 8 || x.ResidentLengthTokens != 128 || !reflect.DeepEqual(x.MemberBlocks, x.ResidentMemberBlocks[:len(x.MemberBlocks)]) || x.Benefit.FreeBytes != int64(len(x.MemberBlocks))*100 {
			t.Fatal("wrong tail provenance/value denominator", x)
		}
	}
	// Mimic the completed physical removal, then obtain a fresh next decision.
	c.Candidates = c.Candidates[:5]
	c.ResidentHashes = c.ResidentHashes[:5]
	c.DeficitBlocks = 2
	c.BenefitCosts = benefitState(t, c)
	d = p.Choose(c)
	ids = nil
	for _, a := range d.Reclaim {
		ids = append(ids, a.BlockID)
	}
	if !reflect.DeepEqual(ids, []int64{4, 3}) {
		t.Fatal("partial removal left holes or wrong next leaf", ids)
	}
	for _, n := range p.nodes {
		if n.frequency != 1 || n.clock != 255 {
			t.Fatal("tail split changed per-hash heat")
		}
	}
}

func TestTailOptionsCannotCrossProtectedOrDuplicateBoundaries(t *testing.T) {
	p := tailPolicy(t)
	c := logicalContext(p, []string{"a", "ab", "abc", "abcd"})
	c.Candidates = append(c.Candidates[:2], c.Candidates[3]) // protected abc remains resident
	c.DeficitBlocks = 4
	c.BenefitCosts = benefitState(t, c)
	d := p.Choose(c)
	if len(d.Reclaim) != 1 || d.BlockID != 3 {
		t.Fatal("crossed protected parent", d)
	}
	c = logicalContext(p, []string{"a", "ab", "abc", "abcd"})
	c.Candidates = append(c.Candidates, PeerCandidate{ID: 4, Hash: "abcd"})
	c.ResidentHashes = append(c.ResidentHashes, "abcd")
	c.DeficitBlocks = 4
	c.BenefitCosts = benefitState(t, c)
	p.Choose(c)
	for _, g := range p.lastSegments {
		if len(g.members) != 1 || len(g.residentMembers) != 1 {
			t.Fatal("duplicate copy inflated tail group", g)
		}
	}
}

func TestTailOptionsIncludeNonPowerDeficitWithoutDuplicateOptions(t *testing.T) {
	p := tailPolicy(t)
	c := logicalContext(p, []string{"a", "ab", "abc", "abcd", "abcde", "abcdef"})
	c.DeficitBlocks = 5
	c.BenefitCosts = benefitState(t, c)
	p.Choose(c)
	var sizes []int
	for _, g := range p.lastSegments {
		sizes = append(sizes, len(g.members))
	}
	if !reflect.DeepEqual(sizes, []int{1, 2, 4, 5}) {
		t.Fatal(sizes)
	}
	defer func() {
		if recover() == nil {
			t.Error("tail policy accepted missing actual deficit")
		}
	}()
	c.DeficitBlocks = 0
	p.Choose(c)
}
