package kv

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func hotPolicy(t *testing.T) *HotPrefixPolicy {
	t.Helper()
	p, err := NewHotPrefixPolicy(HotPrefixConfig{AgingIntervalRequests: 2, AdmissionThreshold: 2, MaxAge: 3, PromotionBlocks: 2}, 16)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestHotPrefixCountsPublishedReuseAndAgesSaturatingClocks(t *testing.T) {
	p := hotPolicy(t)
	p.observe([]string{"a", "ab"}, 0)
	if p.nodes["a"].frequency != 0 {
		t.Fatal("uncomputed request created reuse")
	}
	p.published("a")
	p.published("ab")
	p.observe([]string{"a", "ac"}, 1)
	if p.nodes["a"].frequency != 2 || p.nodes["a"].clock != 2 || p.nodes["ab"].frequency != 1 || p.nodes["ab"].clock != 2 || p.nodes["ac"].frequency != 0 {
		t.Fatal("shared path or request-distance aging changed", p.nodes)
	}
	p.published("a")
	if p.nodes["a"].frequency != 2 {
		t.Fatal("publication double-counted an access")
	}
	for i := 0; i < 600; i++ {
		p.observe([]string{"a"}, 1)
	}
	if p.nodes["a"].frequency != 255 || p.nodes["ab"].clock != 0 {
		t.Fatal("8-bit history did not saturate")
	}
}

func TestHotPrefixEvictsLeafAndWaitsForProtectedDescendants(t *testing.T) {
	p := hotPolicy(t)
	p.remember([]string{"a", "ab"})
	p.remember([]string{"c"})
	for _, h := range []string{"a", "ab", "c"} {
		p.published(h)
	}
	p.nodes["a"].clock = 0 // a has the lowest score, but it is not a leaf
	c := PeerReclaimContext{Candidates: []PeerCandidate{{ID: 1, Hash: "a"}, {ID: 2, Hash: "c"}}, ResidentHashes: []string{"a", "ab", "c"}}
	if got := p.Choose(c); got.Decline || got.BlockID != 2 {
		t.Fatal("reclaimed ancestor of protected child", got)
	}
	c.Candidates = c.Candidates[:1]
	if !p.Choose(c).Decline {
		t.Fatal("must wait when only an ancestor is idle")
	}
	c.ResidentHashes = []string{"a", "a", "ab"}
	if p.Choose(c).Decline || p.Choose(c).Pool != "" {
		t.Fatal("redundant physical ancestor copy should drop without offload")
	}
	c.ResidentHashes = []string{"a"}
	if p.Choose(c).BlockID != 1 || p.Choose(c).Decline {
		t.Fatal("parent did not become eligible after child removal")
	}
}

func TestHotPrefixAdmissionProtectsHotAndPinnedDRAMCopies(t *testing.T) {
	p := hotPolicy(t)
	p.remember([]string{"a"})
	p.remember([]string{"b"})
	p.published("a")
	p.published("b")
	pool := hotPrefixPoolPolicy{p}
	c := PoolAdmissionContext{Hash: "b", CapacityBlocks: 1, UsedBlocks: 0}
	if pool.Admit(c).Accept {
		t.Fatal("below-threshold node admitted")
	}
	p.nodes["b"].frequency = 2
	if !pool.Admit(c).Accept {
		t.Fatal("free space rejected")
	}
	c.UsedBlocks = 1
	if pool.Admit(c).Accept {
		t.Fatal("overwrote pinned/reserved copy")
	}
	c.Candidates = []PoolEvictionCandidate{{Hash: "a"}}
	p.nodes["a"].frequency = 2
	if pool.Admit(c).Accept {
		t.Fatal("equal-hotness churn accepted")
	}
	p.nodes["b"].frequency = 3
	if !pool.Admit(c).Accept || pool.Victim(c.Candidates) != "a" {
		t.Fatal("hotter replacement not selected")
	}
}

func TestHotPrefixPromotionUsesReadyRootsAndKeepsParents(t *testing.T) {
	p := hotPolicy(t)
	p.remember([]string{"a", "ab", "abc"})
	p.remember([]string{"c"})
	p.remember([]string{"d"})
	for h := range p.nodes {
		p.published(h)
	}
	p.nodes["ab"].frequency = 9
	p.nodes["abc"].frequency = 20
	p.nodes["a"].frequency = 1
	p.nodes["c"].frequency = 2
	resident := []string{"a", "c"}
	idle := []PeerCandidate{{ID: 1, Hash: "a"}, {ID: 2, Hash: "c"}}
	plan := p.planPromotions(resident, idle, []string{"ab", "abc"}, 0, 2)
	if !reflect.DeepEqual(plan, []hotPrefixPromotion{{hash: "ab", victim: 2}}) {
		t.Fatal("selected nonroot or replaced parent", plan)
	}
	if len(p.planPromotions(resident, idle, []string{"ab"}, 1, 0)) != 0 {
		t.Fatal("budget ignored")
	}
	if got := p.planPromotions(resident, idle, []string{"ab"}, 1, 1); len(got) != 1 || got[0].victim != -1 {
		t.Fatal("free capacity unnecessarily evicted a block", got)
	}
}

func TestHotPrefixActualReclaimRetriesAndAdmission(t *testing.T) {
	h, a, _ := newPeerHarness(t, false)
	h.f.stores = []*PeerCache{a}
	// Harness also has a CXL pool; single-tier adapter rejects it until scoped.
	for id := range h.f.pools {
		if id != "dram" {
			delete(h.f.pools, id)
		}
	}
	a.access = a.access[:1]
	if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "dram", BackgroundStoreMode: "on_reclaim"}); err != nil {
		t.Fatal(err)
	}
	if err := a.ConfigureHotPrefix(HotPrefixConfig{AgingIntervalRequests: 10, AdmissionThreshold: 1}); err != nil {
		t.Fatal(err)
	}
	seedPeer(t, a, "seed", []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8})
	before := a.hotprefix.policy.requests
	if a.makeSpace(2, nil, "pressure") {
		t.Fatal("store must wait")
	}
	for i := 0; i < 4; i++ {
		a.makeSpace(2, nil, "pressure")
	}
	if reclaimCount(h) != 1 {
		t.Fatal("pending leaf caused redundant ancestor store", reclaimCount(h))
	}
	for len(h.events) > 0 {
		h.next(t)
	}
	a.makeSpace(2, nil, "pressure")
	for len(h.events) > 0 {
		h.next(t)
	}
	if !a.makeSpace(2, nil, "pressure") || reclaimCount(h) != 2 {
		t.Fatal("did not reclaim exactly two blocks")
	}
	if a.hotprefix.policy.requests != before {
		t.Fatal("allocation retry counted request heat")
	}
	assertPeerConservation(t, a)
}

func TestHotPrefixPromotionProtectsParentUntilAdoption(t *testing.T) {
	h, s, _ := newPeerHarness(t, false)
	h.f.stores = []*PeerCache{s}
	delete(h.f.pools, "cxl")
	s.access = s.access[:1]
	m := PeerMechanisms{BackgroundStorePool: "dram", BackgroundStoreMode: "on_reclaim", GroupTransfers: true, ConcurrentRestores: true}
	if err := h.f.ConfigureMechanisms(m); err != nil {
		t.Fatal(err)
	}
	if err := s.ConfigureHotPrefix(HotPrefixConfig{AgingIntervalRequests: 10, AdmissionThreshold: 1, PromotionBlocks: 1, PlannerUS: 7}); err != nil {
		t.Fatal(err)
	}
	tokens := []sim.TokenID{1, 2, 3, 4, 5, 6}
	seedPeer(t, s, "seed", tokens)
	keys := s.hotPrefixKeys(tokens)
	parentID := s.lookup[keys[1]]
	if !s.store(s.Blocks[s.lookup[keys[2]]], "dram", "eviction") {
		t.Fatal("failed to seed actual offload")
	}
	for len(h.events) > 0 {
		h.next(t)
	}
	if err := s.ConfigureEnginePhases(fixedEnginePhases{EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2}, 3}, nil); err != nil {
		t.Fatal(err)
	}
	var end int64
	work := []sim.BatchWork{{Request: &sim.Request{ID: "decode", InputTokens: []sim.TokenID{9}}, PrefixTokens: 1, NewTokens: 1}}
	if ok, err := s.BeginBatch(100, work, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	if s.Blocks[parentID].RefCount != 1 || len(s.hotprefix.parentHolds) != 1 {
		t.Fatal("parent was not held across asynchronous copy")
	}
	if _, ready := s.lookup[keys[2]]; ready {
		t.Fatal("promotion was visible before adoption")
	}
	if s.makeSpace(2, nil, "pressure") {
		t.Fatal("pressure reclaimed a protected promotion parent")
	}
	for h.f.active > 0 {
		phaseNext(t, h)
	}
	if s.Blocks[parentID].RefCount != 1 {
		t.Fatal("physical DMA completion released parent before adoption")
	}
	end = 0
	if ok, err := s.BeginBatch(200, nil, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end == 0 {
		phaseNext(t, h)
	}
	if _, ready := s.lookup[keys[2]]; !ready || s.Blocks[parentID].RefCount != 0 || len(s.hotprefix.parentHolds) != 0 {
		t.Fatal("adoption did not publish/release")
	}
	var service, submit int64
	for _, r := range h.records {
		if r.Name == "hotprefix_planner_service" {
			service = r.Time + r.Duration
		}
		if r.Name == "transfer_submit_begin" && r.Reason == "promotion" {
			submit = r.Time
		}
	}
	if service != 117 || submit < service {
		t.Fatal("declared planner cost bypassed", service, submit)
	}
	assertPeerConservation(t, s)
}
