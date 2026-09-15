package kv

import (
	"fmt"
	"strings"
	"testing"
)

type reclaimCallback func(PeerReclaimContext) PeerDecision

func (p reclaimCallback) Choose(c PeerReclaimContext) PeerDecision { return p(c) }

func TestPeerPolicyCannotAuthorizeProtectedBlockByEditingCandidates(t *testing.T) {
	h, cache, _ := newPeerHarness(t, false)
	seedReclaimCache(t, cache)
	protected := cache.FreeHead
	hash := protected.Hash
	cache.policy = reclaimCallback(func(c PeerReclaimContext) PeerDecision {
		// This ID was excluded by runtime protection. Editing the plugin's
		// snapshot must not also edit the eligibility list used for validation.
		c.Candidates[0].ID = protected.ID
		return PeerDecision{BlockID: protected.ID}
	})
	defer func() {
		failure := recover()
		if failure == nil || !strings.Contains(fmt.Sprint(failure), "non-candidate") {
			t.Fatalf("expected eligibility rejection, got %v", failure)
		}
		if protected.Hash != hash || protected.RefCount != 0 || reclaimCount(h) != 0 || h.f.Pending() != 0 {
			t.Fatal("invalid plugin choice changed protected data or submitted I/O")
		}
		assertPeerConservation(t, cache)
	}()
	cache.makeSpace(1, map[int64]bool{protected.ID: true}, "request")
}

func TestPeerPolicyRetainedSnapshotCannotChangeNextReclaim(t *testing.T) {
	_, cache, _ := newPeerHarness(t, false)
	seedReclaimCache(t, cache)
	var previous []PeerCandidate
	seen := map[int64]bool{}
	cache.policy = reclaimCallback(func(c PeerReclaimContext) PeerDecision {
		// On the second call, the runtime has compacted its remaining list.
		// Mutating a retained first snapshot must not corrupt that list.
		for i := range previous {
			previous[i].ID = -100
		}
		id := c.Candidates[0].ID
		if id < 0 || seen[id] {
			t.Fatal("retained plugin snapshot corrupted remaining candidates")
		}
		seen[id] = true
		previous = c.Candidates
		return PeerDecision{BlockID: id}
	})
	if !cache.makeSpace(3, nil, "request") || len(seen) != 3 {
		t.Fatal("valid custom victim choices did not reclaim three distinct blocks")
	}
	assertPeerConservation(t, cache)
}
