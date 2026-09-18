package kv

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func logicalPolicy(t *testing.T, mode string) *HotPrefixPolicy {
	t.Helper()
	p, err := NewHotPrefixPolicy(HotPrefixConfig{AgingIntervalRequests: 16, AdmissionThreshold: 2,
		HBMEvictionUnit: "logical_segment", HBMScore: mode, LengthReferenceTokens: 16}, 16)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func logicalContext(p *HotPrefixPolicy, chains ...[]string) PeerReclaimContext {
	c := PeerReclaimContext{}
	for _, chain := range chains {
		p.remember(chain)
		for _, hash := range chain {
			p.published(hash)
			c.Candidates = append(c.Candidates, PeerCandidate{ID: int64(len(c.Candidates)), Hash: hash})
			c.ResidentHashes = append(c.ResidentHashes, hash)
		}
	}
	return c
}

func TestLogicalLengthChangesWholeSegmentSelection(t *testing.T) {
	for mode, want := range map[string][]int64{"paper": {5, 4, 3, 2}, "paper_fixed": {1, 0}, "frequency": {1, 0}, "clock": {1, 0}} {
		p := logicalPolicy(t, mode)
		c := logicalContext(p, []string{"a", "ab"}, []string{"c", "cd", "cde", "cdef"})
		for _, h := range []string{"c", "cd", "cde", "cdef"} {
			p.nodes[h].frequency = 2
		}
		d := p.Choose(c)
		var ids []int64
		for _, a := range d.Reclaim {
			ids = append(ids, a.BlockID)
		}
		if !reflect.DeepEqual(ids, want) {
			t.Fatalf("%s: %+v want %v", mode, d, want)
		}
	}
}

func TestLogicalBranchEndpointAndPartialUsePreserveHistory(t *testing.T) {
	for _, boundary := range []string{"branch", "endpoint", "heat"} {
		p := logicalPolicy(t, "paper")
		c := logicalContext(p, []string{"a", "ab", "abc", "abcd"})
		switch boundary {
		case "branch":
			p.remember([]string{"a", "ab", "other"})
		case "endpoint":
			p.observe([]string{"a", "ab"})
		case "heat":
			p.reuse("ab", 0)
		}
		g := p.logicalSegments(c)
		if len(g) != 1 || !reflect.DeepEqual(g[0].hashes, []string{"abcd", "abc"}) {
			t.Fatalf("%s: %+v", boundary, g)
		}
		if p.nodes["abc"].frequency != 1 || p.nodes["abcd"].frequency != 1 {
			t.Fatal("splitting inflated inherited history")
		}
		if boundary == "heat" && p.nodes["ab"].frequency != 2 {
			t.Fatal("partial completed use not retained")
		}
	}
}

func TestLogicalProtectedMissingAndDuplicateCopies(t *testing.T) {
	p := logicalPolicy(t, "paper")
	c := logicalContext(p, []string{"a", "ab", "abc", "abcd"})
	protected := c
	protected.Candidates = protected.Candidates[:3]
	if !p.Choose(protected).Decline {
		t.Fatal("reclaimed ancestor of protected tail")
	}
	missing := c
	missing.Candidates = []PeerCandidate{c.Candidates[0], c.Candidates[2], c.Candidates[3]}
	missing.ResidentHashes = []string{"a", "abc", "abcd"}
	g := p.logicalSegments(missing)
	if len(g) != 1 || len(g[0].members) != 2 {
		t.Fatal("crossed a residency gap", g)
	}
	duplicate := c
	duplicate.Candidates = append(append([]PeerCandidate(nil), c.Candidates...), PeerCandidate{ID: 4, Hash: "a"})
	duplicate.ResidentHashes = append(append([]string(nil), c.ResidentHashes...), "a")
	seen := map[int64]bool{}
	for _, g := range p.logicalSegments(duplicate) {
		for _, a := range g.members {
			if seen[a.BlockID] {
				t.Fatal("double-counted physical capacity")
			}
			seen[a.BlockID] = true
		}
		if g.hashes[0] == "a" && len(g.members) != 1 {
			t.Fatal("duplicate merged with ancestors")
		}
	}
	if len(seen) != 5 {
		t.Fatal("missing legal duplicate/group member", seen)
	}
}

func TestLogicalBlockBoundaryMappingUsesCommonFullPrefix(t *testing.T) {
	_, s, _ := newPeerHarness(t, false)
	a := s.hotPrefixKeys([]sim.TokenID{1, 2, 3, 4, 5})
	b := s.hotPrefixKeys([]sim.TokenID{1, 2, 3, 9, 8})
	if len(a) != 2 || len(b) != 2 || a[0] != b[0] || a[1] == b[1] {
		t.Fatal("intra-block divergence shared a partial physical block")
	}
}

func TestWholeLogicalGroupReclaimsBeyondImmediateDeficit(t *testing.T) {
	h, s, _ := newPeerHarness(t, false)
	h.f.stores = []*PeerCache{s}
	delete(h.f.pools, "cxl")
	s.access = s.access[:1]
	if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "dram", BackgroundStoreMode: "on_reclaim"}); err != nil {
		t.Fatal(err)
	}
	if err := s.ConfigureHotPrefix(HotPrefixConfig{AgingIntervalRequests: 16, AdmissionThreshold: 255, HBMEvictionUnit: "logical_segment"}); err != nil {
		t.Fatal(err)
	}
	seedPeer(t, s, "seed", []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8})
	if !s.makeSpace(1, nil, "pressure") {
		t.Fatal("whole segment did not free capacity")
	}
	groups, drops := 0, 0
	for _, e := range h.records {
		if e.Name == "hotprefix_group_reclaim" {
			groups++
			if len(e.HBMBlocks) != 4 || e.Counters["deficit_blocks"] != 1 || e.Counters["overshoot_blocks"] != 3 {
				t.Fatal("incorrect whole group accounting", e)
			}
		}
		if e.Name == "hbm_drop" {
			drops++
		}
	}
	if groups != 1 || drops != 4 || len(s.ready) != 0 {
		t.Fatal("stopped eviction after one block", groups, drops)
	}
	assertPeerConservation(t, s)
}

func TestInvalidReclaimGroupCannotPartiallyMutateRuntime(t *testing.T) {
	for _, kind := range []string{"protected", "duplicate", "target"} {
		t.Run(kind, func(t *testing.T) {
			h, s, _ := newPeerHarness(t, false)
			seedReclaimCache(t, s)
			protected := s.FreeHead
			s.policy = reclaimCallback(func(c PeerReclaimContext) PeerDecision {
				first := PeerReclaimAction{BlockID: c.Candidates[0].ID}
				last := PeerReclaimAction{BlockID: protected.ID}
				if kind == "duplicate" {
					last = first
				}
				if kind == "target" {
					last = PeerReclaimAction{BlockID: c.Candidates[1].ID, Pool: "nonexistent"}
				}
				return PeerDecision{BlockID: first.BlockID, Reclaim: []PeerReclaimAction{first, last}}
			})
			defer func() {
				if recover() == nil {
					t.Fatal("invalid group accepted")
				}
				if reclaimCount(h) != 0 || h.f.Pending() != 0 || len(s.ready) != 4 {
					t.Fatal("invalid tail action mutated earlier member")
				}
				assertPeerConservation(t, s)
			}()
			s.makeSpace(1, map[int64]bool{protected.ID: true}, "r")
		})
	}
}
