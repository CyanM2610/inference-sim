package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func requestPromotionFixture(t *testing.T) (*peerHarness, *PeerCache, []sim.TokenID) {
	t.Helper()
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 10, TailUS: 2})
	if err := s.EnableRestoreDecisions(); err != nil {
		t.Fatal(err)
	}
	if err := s.EnablePromotionDecisionsWithRetention("request"); err != nil {
		t.Fatal(err)
	}
	return h, s, tokens
}

func adoptPromotion(t *testing.T, h *peerHarness, s *PeerCache) {
	t.Helper()
	var end int64
	s.BeginBatch(100, nil, func(at int64) { end = at })
	for end == 0 || h.f.active > 0 {
		phaseNext(t, h)
	}
	if len(s.pendingTargets) == 0 {
		t.Fatal("fixture must miss first poll")
	}
	end = 0
	s.BeginBatch(200, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
}

func TestRequestPromotionRetentionThroughAdoptionAndRelease(t *testing.T) {
	for _, cancel := range []string{"none", "during_copy", "after_adoption"} {
		t.Run(cancel, func(t *testing.T) {
			h, s, tokens := requestPromotionFixture(t)
			q := sim.DecisionKVQuery{ID: "owner", Input: tokens}
			out := s.PromoteRequestPrefix(q, 2)
			if out.StartedBlocks != 2 || s.FreeBlockCnt != 4 || s.readPending[q.ID] != 2 {
				t.Fatal(out)
			}
			if s.PromotionRetention() != "request" || s.restoreLimits[q.ID] != 2 {
				t.Fatal("missing retention/limit")
			}
			if cancel == "during_copy" {
				s.ClearDeferred(q.ID)
				if s.FreeBlockCnt != 4 || h.f.Snapshot()["cxl"]["read_pins"] != 2 {
					t.Fatal("cancel released pending leases")
				}
			}
			adoptPromotion(t, h, s)
			if len(s.GetCachedBlocks(tokens)) != 2 || h.f.Pending() != 0 || h.f.Snapshot()["cxl"]["read_pins"] != 0 {
				t.Fatal("adoption failed")
			}
			if cancel != "during_copy" {
				if s.FreeBlockCnt != 4 || len(s.holds[q.ID]) != 2 {
					t.Fatal("adoption dropped request ownership")
				}
				if cancel == "after_adoption" {
					s.ClearDeferred(q.ID)
				} else {
					r := &sim.Request{ID: q.ID, InputTokens: tokens}
					if !s.AllocateKVBlocks(r, 4, 5, s.GetCachedBlocks(tokens)) {
						t.Fatal("owner could not consume retained prefix")
					}
					s.ReleaseKVBlocks(r)
				}
			}
			if s.FreeBlockCnt != 6 || len(s.holds) != 0 || len(s.prefillTargets) != 0 || len(s.restoreLimits) != 0 {
				t.Fatal("retention lifecycle leaked")
			}
			assertPeerConservation(t, s)
		})
	}
}

func TestRequestPromotionChecksWholeRangeAgainstOtherReservations(t *testing.T) {
	h, s, tokens := requestPromotionFixture(t)
	running := &sim.Request{ID: "running", InputTokens: []sim.TokenID{30, 31, 32, 33, 34, 35, 36, 37, 38}}
	if !s.AllocateKVBlocks(running, 0, 2, nil) {
		t.Fatal("fixture admission")
	}
	if s.FreeBlockCnt != 5 || s.otherPrefillReservations("reader") != 4 {
		t.Fatal("fixture reservation")
	}
	out := s.PromoteRequestPrefix(sim.DecisionKVQuery{ID: "reader", Input: tokens}, 2)
	if out.StartedBlocks != 0 || out.BlockedBlocks != 2 || h.f.Pending() != 0 || s.FreeBlockCnt != 5 {
		t.Fatal("range rejection stranded a partial native-style load", out)
	}
	if h.f.Snapshot()["cxl"]["read_pins"] != 0 || len(s.holds["reader"]) != 0 {
		t.Fatal("failed allocation acquired leases")
	}
	s.ClearDeferred("reader")
	s.ReleaseKVBlocks(running)
	assertPeerConservation(t, s)
}

func TestRequestPromotionJoinsPartialPrefixWithoutLoadingSuffix(t *testing.T) {
	h, s, tokens := requestPromotionFixture(t)
	first := s.PromoteRequestPrefix(sim.DecisionKVQuery{ID: "owner", Input: tokens}, 1)
	joined := s.PromoteRequestPrefix(sim.DecisionKVQuery{ID: "follower", Input: tokens}, 2)
	if first.StartedBlocks != 1 || joined.PendingBlocks != 1 || joined.BlockedBlocks != 1 || joined.StartedBlocks != 0 || h.f.Pending() != 1 {
		t.Fatal("follower allocated a second transaction before dependency publication", first, joined)
	}
	if len(s.holds["follower"]) != 0 || s.readPending["follower"] != 0 || s.restoreLimits["follower"] != 2 {
		t.Fatal("follower gained fabricated ownership")
	}
	s.ClearDeferred("owner")
	adoptPromotion(t, h, s)
	if len(s.GetCachedBlocks(tokens)) != 1 {
		t.Fatal("wrong prefix published")
	}
	s.ClearDeferred("follower")
	assertPeerConservation(t, s)
}

func TestPromotionProtectsSourceBeforeVictimWriteback(t *testing.T) {
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 10, TailUS: 2})
	s.KVCacheState = NewKVCacheState(1, 2)
	s.fabric.mechanisms.BackgroundStorePool = ""
	seedPeer(t, s, "victim", []sim.TokenID{90, 91})
	s.policy = BuiltinPeerPolicy{Name: "lfu_store", MinFrequency: 1}
	s.access = s.access[1:]
	key := s.hashes(tokens)[0]
	source := h.f.pools["cxl"].entries[key]
	h.f.pools["cxl"].config.CapacityBlocks = 1
	h.f.pools["cxl"].entries = map[string]*peerEntry{key: source}
	if err := s.EnablePromotionDecisions(); err != nil {
		t.Fatal(err)
	}
	out := s.PromoteRequestPrefix(sim.DecisionKVQuery{ID: "reader", Input: tokens}, 1)
	if out.StartedBlocks != 1 || h.f.pools["cxl"].entries[key] != source || source.readers != 1 {
		t.Fatal("victim write replaced the source being promoted", out)
	}
	adoptPromotion(t, h, s)
	if len(s.GetCachedBlocks(tokens)) != 1 || source.readers != 0 {
		t.Fatal("protected source did not survive copy")
	}
	assertPeerConservation(t, s)
}

func TestRequestPromotionProtectsEveryPlannedSourceBeforeWriteback(t *testing.T) {
	h, s, tokens := requestPromotionFixture(t)
	s.KVCacheState = NewKVCacheState(3, 2)
	s.fabric.mechanisms.BackgroundStorePool = ""
	seedPeer(t, s, "victims", []sim.TokenID{90, 91, 92, 93, 94, 95})
	s.policy = BuiltinPeerPolicy{Name: "lfu_store", MinFrequency: 1}
	s.access = s.access[1:]
	keys := s.hashes(tokens)
	pool := h.f.pools["cxl"]
	pool.entries = map[string]*peerEntry{keys[0]: pool.entries[keys[0]], keys[1]: pool.entries[keys[1]]}
	pool.config.CapacityBlocks = 2
	out := s.PromoteRequestPrefix(sim.DecisionKVQuery{ID: "reader", Input: tokens}, 2)
	if out.StartedBlocks != 2 || out.BlockedBlocks != 0 || pool.entries[keys[0]] == nil || pool.entries[keys[1]] == nil {
		t.Fatal("reclaim evicted a later source in the grouped load", out)
	}
	adoptPromotion(t, h, s)
	if len(s.GetCachedBlocks(tokens)) != 2 || len(s.holds["reader"]) != 2 {
		t.Fatal("source range was not retained through adoption")
	}
	s.ClearDeferred("reader")
	assertPeerConservation(t, s)
}
