package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func reclaimCount(h *peerHarness) int {
	n := 0
	for _, r := range h.records {
		if r.Name == "reclaim_decision" {
			n++
		}
	}
	return n
}

func seedReclaimCache(t *testing.T, a *PeerCache) {
	seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8})
	// The final prompt block is not counted by prefix lookup, but is complete.
	// Make all four idle blocks eligible for the STORE policy in this fixture.
	for _, b := range a.Blocks {
		a.frequency[b.Hash] = 1
	}
}

func TestPeerReclaimRetriesReuseOutstandingStores(t *testing.T) {
	h, a, _ := newPeerHarness(t, false)
	seedReclaimCache(t, a)
	if err := h.f.ConfigureMechanisms(PeerMechanisms{CompletionUS: 7, ControlWorkers: 1}); err != nil {
		t.Fatal(err)
	}
	if a.makeSpace(2, nil, "waiting") || reclaimCount(h) != 2 {
		t.Fatal("must submit exactly two victims and wait for publication")
	}
	// Retrying before DMA completion must not create more victim writes.
	for i := 0; i < 3; i++ {
		if a.makeSpace(2, nil, "waiting") || reclaimCount(h) != 2 {
			t.Fatalf("retry appended victims: got %d, want 2", reclaimCount(h))
		}
	}
	for len(h.events) > 0 {
		h.next(t)
		ready := a.makeSpace(2, nil, "waiting")
		if reclaimCount(h) != 2 {
			t.Fatalf("partial completion appended victims: got %d, want 2", reclaimCount(h))
		}
		if ready != (len(a.storePins) == 0) {
			t.Fatal("pending STORE space became allocatable before publication")
		}
		assertPeerConservation(t, a)
	}
	q := &sim.Request{ID: "waiting", InputTokens: []sim.TokenID{11, 12, 13, 14}}
	if !a.AllocateKVBlocks(q, 0, 4, nil) {
		t.Fatal("request did not admit after its two victims completed")
	}
	a.ReleaseKVBlocks(q)
	assertPeerConservation(t, a)
}

func TestPeerReclaimPromisesFollowCurrentPhysicalReferences(t *testing.T) {
	for _, mode := range []string{"grown_demand", "live_request", "protected_hit", "duplicate_store_pin"} {
		t.Run(mode, func(t *testing.T) {
			h, a, _ := newPeerHarness(t, false)
			seedReclaimCache(t, a)
			if a.makeSpace(1, nil, "first") {
				t.Fatal("expected asynchronous reclaim")
			}
			var victim *KVBlock
			for id := range a.storePins {
				victim = a.Blocks[id]
			}
			need := int64(1)
			protect := map[int64]bool{}
			switch mode {
			case "grown_demand":
				need = 2
			case "live_request":
				a.pin(victim)
			case "protected_hit":
				protect[victim.ID] = true
			case "duplicate_store_pin":
				if !a.store(victim, "dram", "joined") {
					t.Fatal("failed to join outstanding copy")
				}
				need = 2 // Two STORE refs are still only one physical block.
			}
			if a.makeSpace(need, protect, "second") || reclaimCount(h) != 2 {
				t.Fatalf("must add exactly the uncovered deficit; victims=%d", reclaimCount(h))
			}
			if a.makeSpace(need, protect, "second") || reclaimCount(h) != 2 {
				t.Fatal("repeated retry appended another victim")
			}
			for len(h.events) > 0 {
				h.next(t)
			}
			if mode == "live_request" {
				if victim.RefCount != 1 || victim.Hash == "" {
					t.Fatal("STORE completion invalidated an adopted source")
				}
				a.unpin(victim)
			}
			assertPeerConservation(t, a)
		})
	}
}
