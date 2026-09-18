package kv

import (
	"reflect"
	"testing"
)

func TestHotPrefixHBMScoreAblationsKeepStableTies(t *testing.T) {
	for mode, want := range map[string]int64{"paper": 2, "frequency": 2, "clock": 1, "lru": 1} {
		p, err := NewHotPrefixPolicy(HotPrefixConfig{AgingIntervalRequests: 16, AdmissionThreshold: 2, HBMScore: mode}, 16)
		if err != nil {
			t.Fatal(err)
		}
		p.remember([]string{"older"})
		p.remember([]string{"fresh"})
		p.published("older")
		p.published("fresh")
		p.nodes["older"].frequency = 2
		p.nodes["older"].clock = 248
		p.nodes["fresh"].frequency = 1
		p.nodes["fresh"].clock = 255
		c := PeerReclaimContext{Candidates: []PeerCandidate{{ID: 1, Hash: "older"}, {ID: 2, Hash: "fresh"}}, ResidentHashes: []string{"older", "fresh"}}
		if got := p.Choose(c); got.Decline || got.BlockID != want {
			t.Fatalf("%s: got %+v want %d", mode, got, want)
		}
		p.nodes["older"].frequency = 1
		p.nodes["older"].clock = 255
		if p.Choose(c).BlockID != 1 {
			t.Fatal("tie changed native LRU order", mode)
		}
	}
}

func TestHotPrefixAllIdleIsIndependentOfLeafFiltering(t *testing.T) {
	p, err := NewHotPrefixPolicy(HotPrefixConfig{AgingIntervalRequests: 16, AdmissionThreshold: 2, HBMScore: "lru"}, 16)
	if err != nil {
		t.Fatal(err)
	}
	p.remember([]string{"parent", "child"})
	p.published("parent")
	p.published("child")
	c := PeerReclaimContext{Candidates: []PeerCandidate{{ID: 1, Hash: "parent"}, {ID: 2, Hash: "child"}}, ResidentHashes: []string{"parent", "child"}}
	if p.Choose(c).BlockID != 2 {
		t.Fatal("leaf comparator reclaimed ancestor")
	}
	p.config.HBMCandidates = "all_idle"
	if p.Choose(c).BlockID != 1 {
		t.Fatal("ordinary idle LRU retained the leaf filter")
	}
}

func TestHotPrefixChoiceDiagnosticsDoNotAlterPolicyState(t *testing.T) {
	p := hotPolicy(t)
	p.remember([]string{"parent", "child"})
	p.published("parent")
	p.published("child")
	c := PeerReclaimContext{Candidates: []PeerCandidate{{ID: 1, Hash: "parent"}, {ID: 2, Hash: "child"}}, ResidentHashes: []string{"parent", "child"}}
	want := p.Choose(c)
	p.config.Diagnostics = &HotPrefixDiagnosticsConfig{TraceCandidates: true, MeasureCPU: true}
	p.diagnostics = &HotPrefixDiagnostics{MeasureCPU: true, CPU: map[string]HotPrefixCPU{}}
	before := *p.nodes["child"]
	got := p.Choose(c)
	if !reflect.DeepEqual(got, want) || *p.nodes["child"] != before || p.requests != 0 {
		t.Fatal("observer changed decision or heat")
	}
	d := p.diagnostics
	if d.Choices != 1 || d.CandidateVisits != 2 || d.CPU["hbm_choose"].Calls != 1 || len(d.Records) != 1 {
		t.Fatal("missing measured decision", d)
	}
	r := d.Records[0]
	if len(r.Candidates) != 2 || r.Candidates[0].Eligible || !r.Candidates[1].Eligible || r.Candidates[1].LengthTokens != 16 || r.SelectedBlock != 2 {
		t.Fatal("incorrect candidate snapshot", r)
	}
}

func TestHotPrefixRejectsUnknownAndMixedAblations(t *testing.T) {
	for _, c := range []HotPrefixConfig{
		{AgingIntervalRequests: 16, HBMScore: "made_up"},
		{AgingIntervalRequests: 16, HBMCandidates: "unprotected"},
		{AgingIntervalRequests: 16, HBMScore: "clock", PromotionBlocks: 1},
	} {
		if c.Validate() == nil {
			t.Fatal("accepted invalid experiment", c)
		}
	}
}
