package kv

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func cacheRow(t *testing.T, s *PeerCache, id string) CacheRequestMetrics {
	t.Helper()
	for _, r := range s.CacheRequestStatistics() {
		if r.Request == id {
			return r
		}
	}
	t.Fatal("missing metric row", id)
	return CacheRequestMetrics{}
}

func metricDRAMFixture(t *testing.T) (*peerHarness, *PeerCache, *sim.Request) {
	h, s, r := restoreDecisionFixture(t)
	// The existing physical phase fixture names its memory pool cxl. Select
	// that same pool as local DRAM without changing transfer execution/timing.
	s.fabric.pools["dram"] = s.fabric.pools["cxl"]
	s.fabric.pools["dram"].config.ID = "dram"
	delete(s.fabric.pools, "cxl")
	s.fabric.mechanisms.BackgroundStorePool = ""
	for _, a := range s.access {
		if a.Pool == "cxl" {
			a.Pool = "dram"
			s.access = []PeerAccess{a}
			break
		}
	}
	return h, s, r
}

func TestCacheMetricsDRAMAvailabilityIsNotReuse(t *testing.T) {
	for _, limit := range []int64{0, 1, 2} {
		h, s, r := metricDRAMFixture(t)
		s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: limit}})
		ok := s.AllocateKVBlocks(r, 0, 4, nil)
		before := cacheRow(t, s, r.ID)
		if before.HBMHitTokens != 0 || before.DRAMHitTokens != 3 || before.ReusedTokens != 0 || before.ReuseObserved {
			t.Fatalf("availability must ignore policy cap, not count use: %+v", before)
		}
		if limit > 0 {
			if ok {
				t.Fatal("pending load admitted")
			}
			for i := 0; i < 5; i++ {
				s.AllocateKVBlocks(r, 0, 4, nil)
				s.GetRequestCachedPrefix(r)
			}
			if !reflect.DeepEqual(before, cacheRow(t, s, r.ID)) {
				t.Fatal("retry changed counters")
			}
			finishRestoreStep(t, h, s)
			if cacheRow(t, s, r.ID).ReusedTokens != 0 {
				t.Fatal("DMA/adoption counted as use")
			}
			prefix := s.GetRequestCachedPrefix(r)
			if !s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) {
				t.Fatal("restore admission failed")
			}
		}
		if cacheRow(t, s, r.ID).ReuseObserved {
			t.Fatal("allocation counted as completed compute")
		}
		r.ProgressIndex = 4
		s.MirrorToCPU([]*sim.Request{r})
		row := cacheRow(t, s, r.ID)
		want := min(2*limit, int64(3))
		if !row.ReuseObserved || row.DRAMReusedTokens != want || row.DRAMDemandReusedTokens != want || row.DRAMInitialHitReusedTokens != want || row.ReusedTokens != want || row.HBMReusedTokens != 0 {
			t.Fatalf("limit=%d: %+v", limit, row)
		}
		s.MirrorToCPU([]*sim.Request{r})
		s.ReleaseKVBlocks(r)
		// Recovery and repeated completion callbacks cannot inflate first-use counts.
		r.ProgressIndex = 0
		prefix := s.GetRequestCachedPrefix(r)
		s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: 0}})
		s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks)
		r.ProgressIndex = 4
		s.MirrorToCPU([]*sim.Request{r})
		if !reflect.DeepEqual(row, cacheRow(t, s, r.ID)) {
			t.Fatal("recovery inflated reuse")
		}
		s.ReleaseKVBlocks(r)
	}
}

func TestCacheMetricsHBMAndQueriesAreSeparate(t *testing.T) {
	_, s, _ := newPeerHarness(t, false)
	tokens := []sim.TokenID{1, 2, 3}
	seedPeer(t, s, "warm", tokens)
	before := s.CacheRequestStatistics()
	s.EnableHashWorkAccounting()
	for i := 0; i < 6; i++ {
		s.GetCachedBlocks(tokens)
		s.DecisionState([]sim.DecisionKVQuery{{ID: "query", Input: tokens}})
	}
	if !reflect.DeepEqual(before, s.CacheRequestStatistics()) {
		t.Fatal("query changed metrics")
	}
	s.TakeHashWork()
	s.observeCacheLookup("reader", tokens)
	if s.TakeHashWork().Calls != 0 {
		t.Fatal("metric observer charged policy hashes")
	}
	r := &sim.Request{ID: "reader", InputTokens: tokens}
	p := s.GetRequestCachedPrefix(r)
	if !s.AllocateKVBlocks(r, p.Tokens, 3, p.Blocks) {
		t.Fatal("HBM admission failed")
	}
	r.ProgressIndex = 3
	s.MirrorToCPU([]*sim.Request{r})
	row := cacheRow(t, s, r.ID)
	if row.HBMHitTokens != 2 || row.HBMReusedTokens != 2 || row.DRAMHitTokens != 0 || row.ReusedTokens != 2 {
		t.Fatal(row)
	}
	agg := SummarizeCacheRequests(s.CacheRequestStatistics())
	if *agg.HBMDirectHitRate != 2.0/6 || *agg.HBMRequestHitRate != .5 || agg.DRAMInitialHitReuseFraction != nil {
		t.Fatal(agg)
	}
}

func TestCacheMetricsCancelledAdmissionDoesNotReuse(t *testing.T) {
	h, s, r := metricDRAMFixture(t)
	s.AllocateKVBlocks(r, 0, 4, nil)
	finishRestoreStep(t, h, s)
	p := s.GetRequestCachedPrefix(r)
	if !s.AllocateKVBlocks(r, p.Tokens, 4, p.Blocks) {
		t.Fatal("admission failed")
	}
	s.ReleaseKVBlocks(r) // cancelled before compute
	s.MirrorToCPU([]*sim.Request{r})
	if row := cacheRow(t, s, r.ID); row.ReuseObserved || row.ReusedTokens != 0 {
		t.Fatal(row)
	}
}

func TestCacheMetricsPromotionAndSharedLoad(t *testing.T) {
	h, s, r := metricDRAMFixture(t)
	if err := s.EnablePromotionDecisionsWithRetention("request"); err != nil {
		t.Fatal(err)
	}
	q := sim.DecisionKVQuery{ID: r.ID, Input: r.InputTokens}
	out := s.PromoteRequestPrefix(q, 1)
	if out.StartedBlocks != 1 {
		t.Fatal(out)
	}
	other := &sim.Request{ID: "join", InputTokens: r.InputTokens}
	s.AllocateKVBlocks(other, 0, 4, nil) // observes before load adoption
	finishRestoreStep(t, h, s)
	p := s.GetRequestCachedPrefix(r)
	if !s.AllocateKVBlocks(r, p.Tokens, 4, p.Blocks) {
		t.Fatal("promotion admission")
	}
	r.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{r})
	s.ReleaseKVBlocks(r)
	row := cacheRow(t, s, r.ID)
	if row.DRAMPromotionReusedTokens != 2 || row.DRAMReusedTokens != 2 || row.HBMReusedTokens != 0 {
		t.Fatal(row)
	}
	s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: other.ID, MaxPrefixBlocks: 0}})
	p = s.GetRequestCachedPrefix(other)
	if !s.AllocateKVBlocks(other, p.Tokens, 4, p.Blocks) {
		t.Fatal("shared admission")
	}
	other.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{other})
	s.ReleaseKVBlocks(other)
	row = cacheRow(t, s, other.ID)
	if row.DRAMSharedLoadReusedTokens != 2 || row.DRAMPromotionReusedTokens != 2 {
		t.Fatal(row)
	}
	// A later request sees the same loaded page already in HBM: direct hit,
	// with its prefetch provenance retained, without a second DRAM reuse.
	late := &sim.Request{ID: "late", InputTokens: r.InputTokens}
	p = s.GetRequestCachedPrefix(late)
	s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: late.ID, MaxPrefixBlocks: 0}})
	if !s.AllocateKVBlocks(late, p.Tokens, 4, p.Blocks) {
		t.Fatal("late admission")
	}
	late.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{late})
	row = cacheRow(t, s, late.ID)
	if row.HBMHitTokens != 2 || row.HBMPrefetchedReusedTokens != 2 || row.DRAMReusedTokens != 0 {
		t.Fatal(row)
	}
}

func TestCacheMetricsPendingAndLateProducer(t *testing.T) {
	h, s, r := metricDRAMFixture(t)
	keys := s.fullPrefixHashes(r.InputTokens)
	entry := s.fabric.pools["dram"].entries[keys[0]]
	entry.ready = false
	if s.AllocateKVBlocks(r, 0, 4, nil) {
		t.Fatal("pending store admitted")
	}
	row := cacheRow(t, s, r.ID)
	if !row.LookupPendingStore || row.DRAMHitTokens != 0 || row.ReusedTokens != 0 {
		t.Fatal(row)
	}
	entry.ready = true
	s.AllocateKVBlocks(r, 0, 4, nil)
	finishRestoreStep(t, h, s)
	p := s.GetRequestCachedPrefix(r)
	if !s.AllocateKVBlocks(r, p.Tokens, 4, p.Blocks) {
		t.Fatal("ready store admission")
	}
	r.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{r})
	row = cacheRow(t, s, r.ID)
	if row.DRAMHitTokens != 0 || row.DRAMReusedTokens != 3 || row.DRAMInitialHitReusedTokens != 0 {
		t.Fatal(row)
	}

	_, s, _ = newPeerHarness(t, false)
	tokens := []sim.TokenID{7, 8, 9}
	s.observeCacheLookup("waiting", tokens)
	seedPeer(t, s, "producer", tokens)
	w := &sim.Request{ID: "waiting", InputTokens: tokens}
	p = s.GetRequestCachedPrefix(w)
	s.AllocateKVBlocks(w, p.Tokens, 3, p.Blocks)
	w.ProgressIndex = 3
	s.MirrorToCPU([]*sim.Request{w})
	row = cacheRow(t, s, w.ID)
	if row.ReadyAfterWaitReusedTokens != 2 || row.HBMHitTokens != 0 || row.HBMReusedTokens != 0 {
		t.Fatal(row)
	}
}

func TestCacheMetricsBatchSharing(t *testing.T) {
	_, s, a, b := batchPrefixFixture(t)
	_, computed := formPrefixPair(s, a, b, nil)
	if cacheRow(t, s, b.ID).ReusedTokens != 0 {
		t.Fatal("unexecuted donor counted")
	}
	s.CompletePrefixBatch()
	a.ProgressIndex, b.ProgressIndex = computed[a.ID], computed[b.ID]
	s.MirrorToCPU([]*sim.Request{a, b})
	r := cacheRow(t, s, b.ID)
	if r.BatchSharedReusedTokens != 2 || r.HBMReusedTokens != 0 || r.ReusedTokens != 2 {
		t.Fatal(r)
	}
}

func TestCacheMetricsMixedPrefixDoesNotDoubleCountDRAMCopy(t *testing.T) {
	h, s, warm := metricDRAMFixture(t)
	s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: warm.ID, MaxPrefixBlocks: 1}})
	s.AllocateKVBlocks(warm, 0, 4, nil)
	finishRestoreStep(t, h, s)
	s.ClearDeferred(warm.ID)
	r := &sim.Request{ID: "mixed", InputTokens: warm.InputTokens}
	p := s.GetRequestCachedPrefix(r)
	if s.AllocateKVBlocks(r, p.Tokens, 4, p.Blocks) {
		t.Fatal("missing DRAM suffix did not defer")
	}
	row := cacheRow(t, s, r.ID)
	if row.HBMHitTokens != 2 || row.DRAMHitTokens != 1 {
		t.Fatal("duplicate DRAM copy or replay double counted", row)
	}
	finishRestoreStep(t, h, s)
	p = s.GetRequestCachedPrefix(r)
	if !s.AllocateKVBlocks(r, p.Tokens, 4, p.Blocks) {
		t.Fatal("mixed admission")
	}
	r.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{r})
	row = cacheRow(t, s, r.ID)
	if row.HBMReusedTokens != 2 || row.DRAMReusedTokens != 1 || row.ReusedTokens != 3 {
		t.Fatal(row)
	}
	s.ReleaseKVBlocks(r)
	// Force both physical slots to be reused by computation, then query the new
	// prefix. The previous load origin must not survive block invalidation.
	s.policy = BuiltinPeerPolicy{Name: "lru_drop"}
	seedPeer(t, s, "replacement", []sim.TokenID{91, 92, 93, 94})
	late := &sim.Request{ID: "replacement-reader", InputTokens: []sim.TokenID{91, 92, 93, 94}}
	s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: late.ID, MaxPrefixBlocks: 0}})
	p = s.GetRequestCachedPrefix(late)
	if !s.AllocateKVBlocks(late, p.Tokens, 4, p.Blocks) {
		t.Fatal("replacement admission")
	}
	late.ProgressIndex = 4
	s.MirrorToCPU([]*sim.Request{late})
	row = cacheRow(t, s, late.ID)
	if row.HBMReusedTokens != 2 || row.DRAMReusedTokens != 0 || row.HBMPrefetchedReusedTokens != 0 {
		t.Fatal(row)
	}
}
