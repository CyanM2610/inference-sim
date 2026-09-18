package kv

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func shadowFixture(t *testing.T, ttl *int64) (*peerHarness, *PeerCache) {
	t.Helper()
	h, s, _ := newPeerHarness(t, false)
	h.f.stores = []*PeerCache{s}
	delete(h.f.pools, "cxl")
	s.access = s.access[:1]
	if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "dram", BackgroundStoreMode: "on_reclaim"}); err != nil {
		t.Fatal(err)
	}
	if err := s.ConfigureHotPrefix(HotPrefixConfig{AgingIntervalRequests: 16, AdmissionThreshold: 2, ShadowTTLUS: ttl}); err != nil {
		t.Fatal(err)
	}
	return h, s
}

func TestHotPrefixLateActualReuseCommitsOnce(t *testing.T) {
	_, s := shadowFixture(t, nil)
	tokens := []sim.TokenID{1, 2, 3, 4, 5, 6, 7}
	producer := &sim.Request{ID: "producer", InputTokens: tokens}
	if !s.AllocateKVBlocks(producer, 0, 7, nil) {
		t.Fatal("producer")
	}
	consumer := &sim.Request{ID: "consumer", InputTokens: tokens}
	if s.AllocateKVBlocks(consumer, 0, 2, nil) {
		t.Fatal("consumer must wait")
	}
	producer.ProgressIndex = 7
	s.MirrorToCPU([]*sim.Request{producer})
	s.ReleaseKVBlocks(producer)
	p := s.GetRequestCachedPrefix(consumer)
	if p.Tokens != 6 || !s.AllocateKVBlocks(consumer, p.Tokens, 7, p.Blocks) {
		t.Fatal("late reuse")
	}
	keys := s.hotPrefixKeys(tokens)
	for _, k := range keys {
		if s.hotprefix.policy.nodes[k].frequency != 1 {
			t.Fatal("allocation counted before completion")
		}
	}
	consumer.ProgressIndex = 7
	s.MirrorToCPU([]*sim.Request{consumer})
	s.MirrorToCPU([]*sim.Request{consumer})
	s.ReleaseKVBlocks(consumer)
	// Simulate recovery by this same request: neither reuse nor publication
	// should turn it into another independent popularity observation.
	consumer.ProgressIndex = 0
	p = s.GetRequestCachedPrefix(consumer)
	if !s.AllocateKVBlocks(consumer, p.Tokens, 7, p.Blocks) {
		t.Fatal("recovery")
	}
	consumer.ProgressIndex = 7
	s.MirrorToCPU([]*sim.Request{consumer})
	for _, k := range keys {
		if s.hotprefix.policy.nodes[k].frequency != 2 {
			t.Fatal("missed late reuse or duplicated recovery")
		}
	}
	s.ReleaseKVBlocks(consumer)
	assertPeerConservation(t, s)
}

func TestHotPrefixDoesNotCountUnusedTailOrUnexecutedAllocation(t *testing.T) {
	_, s := shadowFixture(t, nil)
	tokens := []sim.TokenID{10, 11, 12, 13}
	seedPeer(t, s, "seed", tokens)
	keys := s.hotPrefixKeys(tokens)
	r := &sim.Request{ID: "aborted", InputTokens: tokens}
	p := s.GetRequestCachedPrefix(r)
	if p.Tokens != 2 || !s.AllocateKVBlocks(r, p.Tokens, 4, p.Blocks) {
		t.Fatal("tail setup")
	}
	s.ReleaseKVBlocks(r)
	s.MirrorToCPU([]*sim.Request{r})
	if s.hotprefix.policy.nodes[keys[0]].frequency != 1 {
		t.Fatal("unexecuted allocation counted")
	}
	r = &sim.Request{ID: "finished", InputTokens: tokens}
	p = s.GetRequestCachedPrefix(r)
	if !s.AllocateKVBlocks(r, p.Tokens, 4, p.Blocks) {
		t.Fatal("allocation")
	}
	r.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{r})
	if s.hotprefix.policy.nodes[keys[0]].frequency != 2 || s.hotprefix.policy.nodes[keys[1]].frequency != 1 {
		t.Fatal("READY-only tail inflated heat")
	}
	s.ReleaseKVBlocks(r)
	assertPeerConservation(t, s)
}

func TestHotPrefixShadowTimelyRecomputeBecomesAdmissible(t *testing.T) {
	ttl := int64(100)
	h, s := shadowFixture(t, &ttl)
	tokens := []sim.TokenID{1, 2, 3}
	seedPeer(t, s, "first", tokens)
	key := s.hotPrefixKeys(tokens)[0]
	if !s.makeSpace(4, nil, "pressure") || len(s.hotprefix.shadows) != 1 || len(h.f.pools["dram"].entries) != 0 {
		t.Fatal("frequency rejection did not leave metadata-only shadow")
	}
	s.SetClock(99)
	r := &sim.Request{ID: "second", InputTokens: tokens}
	if !s.AllocateKVBlocks(r, 0, 3, nil) {
		t.Fatal("recompute allocation")
	}
	if s.hotprefix.policy.nodes[key].frequency != 1 {
		t.Fatal("shadow counted without execution")
	}
	// A match within TTL is retained for this request even when it completes late.
	s.SetClock(100)
	if len(s.hotprefix.shadows) != 0 || s.hotprefix.policy.nodes[key].frequency != 0 {
		t.Fatal("TTL boundary")
	}
	r.ProgressIndex = 3
	s.MirrorToCPU([]*sim.Request{r})
	s.MirrorToCPU([]*sim.Request{r})
	if s.hotprefix.policy.nodes[key].frequency != 2 || s.hotprefix.policy.nodes[key].clock != 255 {
		t.Fatal("timely shadow reference lost or double counted")
	}
	if cacheRow(t, s, r.ID).ReusedTokens != 0 {
		t.Fatal("metadata-only hit fabricated KV reuse")
	}
	s.ReleaseKVBlocks(r)
	if s.makeSpace(4, nil, "pressure-again") {
		t.Fatal("frequency two should STORE and wait")
	}
	for len(h.events) > 0 {
		h.next(t)
	}
	if !s.makeSpace(4, nil, "after-store") || !h.f.pools["dram"].entries[key].ready {
		t.Fatal("continued frequency failed DRAM admission")
	}
	assertPeerConservation(t, s)
}

func TestHotPrefixShadowExpiryAndDisabledResetHeat(t *testing.T) {
	for _, ttl := range []int64{0, 100} {
		t.Run(fmt.Sprint(ttl), func(t *testing.T) {
			_, s := shadowFixture(t, &ttl)
			tokens := []sim.TokenID{1, 2, 3}
			seedPeer(t, s, "first", tokens)
			key := s.hotPrefixKeys(tokens)[0]
			if !s.makeSpace(4, nil, "pressure") {
				t.Fatal("drop")
			}
			s.SetClock(100)
			if len(s.hotprefix.shadows) != 0 || s.hotprefix.policy.nodes[key].frequency != 0 {
				t.Fatal("shadow retained expired heat")
			}
			seedPeer(t, s, "late", tokens)
			if s.hotprefix.policy.nodes[key].frequency != 1 {
				t.Fatal("late reference resurrected expired history")
			}
		})
	}
}

func TestHotPrefixPartialRestoredTailAndDMAAreNotDoubleCounted(t *testing.T) {
	h, s := shadowFixture(t, nil)
	tokens := []sim.TokenID{20, 21, 22, 23}
	seedPeer(t, s, "seed", tokens)
	keys := s.hotPrefixKeys(tokens)
	r := &sim.Request{ID: "restore", InputTokens: tokens}
	// Model an adopted full-block restore receipt: the final block provides
	// one reusable token, while the last token is replayed for logits.
	s.observeHotPrefix(r)
	blocks := []int64{s.lookup[keys[0]], s.lookup[keys[1]]}
	s.stageHotPrefixReuse(r, 3, blocks, true)
	for _, id := range blocks {
		s.publishReadyCopy(s.Blocks[id])
	}
	if s.hotprefix.policy.nodes[keys[1]].frequency != 1 {
		t.Fatal("DMA publication counted as reuse")
	}
	s.completeHotPrefixReuse(r)
	s.completeHotPrefixReuse(r)
	var reused int64
	for _, e := range h.records {
		if e.Name == "hotprefix_reuse" && e.Request == r.ID {
			reused += e.Counters["reused_tokens"]
		}
	}
	if reused != 3 || s.hotprefix.policy.nodes[keys[0]].frequency != 2 || s.hotprefix.policy.nodes[keys[1]].frequency != 2 {
		t.Fatal("partial tail or duplicate completion", reused)
	}
}

func TestHotPrefixShadowDoesNotReplaceExistingKVAndConfigIsBounded(t *testing.T) {
	_, s := shadowFixture(t, nil)
	tokens := []sim.TokenID{1, 2, 3}
	seedPeer(t, s, "seed", tokens)
	key := s.hotPrefixKeys(tokens)[0]
	s.shadowHotPrefixDrop(key, "duplicate")
	if len(s.hotprefix.shadows) != 0 || s.hotprefix.shadowTTL != 60_000_000 {
		t.Fatal("existing KV treated as shadow or default TTL wrong")
	}
	seedPeer(t, s, "duplicate", tokens)
	s.invalidate(s.Blocks[s.lookup[key]])
	s.shadowHotPrefixDrop(key, "latest-copy-drop")
	if len(s.hotprefix.shadows) != 0 {
		t.Fatal("older physical copy lost from shadow ownership check")
	}
	negative := int64(-1)
	if _, err := NewHotPrefixPolicy(HotPrefixConfig{AgingIntervalRequests: 1, ShadowTTLUS: &negative}, 16); err == nil {
		t.Fatal("negative TTL accepted")
	}
}

func TestHotPrefixLowFrequencyDropKeepsShadowWhenPoolUnavailable(t *testing.T) {
	h, s := shadowFixture(t, nil)
	tokens := []sim.TokenID{1, 2, 3}
	seedPeer(t, s, "seed", tokens)
	key := s.hotPrefixKeys(tokens)[0]
	// A full pool of pending targets offers no STORE destination. This must
	// not bypass retention of an otherwise below-threshold victim's history.
	h.f.pools["dram"].config.CapacityBlocks = 1
	h.f.pools["dram"].entries["pending-target"] = &peerEntry{}
	if !s.makeSpace(4, nil, "pressure") || len(s.hotprefix.shadows) != 1 {
		t.Fatal("unavailable pool bypassed shadow retention")
	}
	if _, ok := s.hotprefix.shadows[key]; !ok {
		t.Fatal("wrong shadow identity")
	}
}

func TestHotPrefixPromotionDropKeepsColdShadowWithoutHeatingDMA(t *testing.T) {
	h, s := shadowFixture(t, nil)
	hot := []sim.TokenID{91, 92, 93}
	seedPeer(t, s, "hot", hot)
	key := s.hotPrefixKeys(hot)[0]
	s.hotprefix.policy.nodes[key].frequency = 2
	if !s.store(s.Blocks[s.lookup[key]], "dram", "seed-store") {
		t.Fatal("seed offload")
	}
	for len(h.events) > 0 {
		h.next(t)
	}
	cold := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8}
	seedPeer(t, s, "cold", cold)
	tail := s.hotPrefixKeys(cold)[3]
	s.hotprefix.policy.config.PromotionBlocks = 1
	s.executeHotPrefixPromotion()
	if _, ok := s.hotprefix.shadows[tail]; !ok {
		t.Fatal("promotion lost below-threshold victim history")
	}
	for len(h.events) > 0 {
		h.next(t)
	}
	if s.hotprefix.policy.nodes[key].frequency != 2 || h.f.pools["dram"].entries[tail] != nil {
		t.Fatal("promotion fabricated reuse or cold writeback")
	}
	assertPeerConservation(t, s)
}
