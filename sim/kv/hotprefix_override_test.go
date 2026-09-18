package kv

import "testing"

func TestOneShotReclaimUsesLegalSnapshotThenReturnsToLRU(t *testing.T) {
	p := logicalPolicy(t, "lru")
	p.config.Diagnostics = &HotPrefixDiagnosticsConfig{TraceCandidates: true}
	p.diagnostics = &HotPrefixDiagnostics{}
	c := logicalContext(p, []string{"a", "ab"}, []string{"b", "bc"})
	g := p.logicalSegments(c)
	p.config.OneShotReclaim = &OneShotReclaim{Sequence: 2, CandidatesSHA256: logicalCandidateDigest(g, c.DeficitBlocks), MemberBlocks: []int64{3, 2}, MemberHashes: []string{"bc", "b"}}
	if p.Choose(c).BlockID != 1 {
		t.Fatal("changed prefix")
	}
	if p.Choose(c).BlockID != 3 || !p.diagnostics.Records[1].OfflineOverride {
		t.Fatal("did not apply exact intervention")
	}
	if p.Choose(c).BlockID != 1 || p.diagnostics.Records[2].OfflineOverride {
		t.Fatal("override persisted")
	}
}

func TestOneShotReclaimRejectsChangedCandidates(t *testing.T) {
	p := logicalPolicy(t, "lru")
	p.config.Diagnostics = &HotPrefixDiagnosticsConfig{TraceCandidates: true}
	p.diagnostics = &HotPrefixDiagnostics{}
	c := logicalContext(p, []string{"a", "ab"}, []string{"b", "bc"})
	g := p.logicalSegments(c)
	p.config.OneShotReclaim = &OneShotReclaim{Sequence: 1, CandidatesSHA256: logicalCandidateDigest(g, c.DeficitBlocks), MemberBlocks: []int64{3, 2}, MemberHashes: []string{"bc", "b"}}
	c.Candidates = c.Candidates[:3]
	defer func() {
		if recover() == nil {
			t.Fatal("stale intervention was accepted")
		}
	}()
	p.Choose(c)
}
