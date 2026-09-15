package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"math"
	"testing"
)

func TestBatchPrefixActualTimeoutKeepsSubmittedWorkAndNoCancelledOutput(t *testing.T) {
	for _, cancelAll := range []bool{false, true} {
		h, cache, a, b := batchPrefixFixture(t)
		cfg := sim.SimConfig{Horizon: math.MaxInt64, KVCacheConfig: sim.NewKVCacheConfig(6, 2, 0, 0, 0, 0), BatchConfig: sim.NewBatchConfig(2, 4, 2), BatchCompletionEvents: true, Seed: 42}
		runtime, err := sim.NewSimulator(cfg, cache, decisionServiceLatency{})
		if err != nil {
			t.Fatal(err)
		}
		cache.BindEvents(runtime.Schedule, runtime.ScheduleStepIfIdle)
		a.Deadline = 5
		if cancelAll {
			b.Deadline = 6
		}
		runtime.InjectArrival(a)
		runtime.InjectArrival(b)
		cancelObserved := false
		for n := 0; runtime.HasPendingEvents(); n++ {
			if n > 500 {
				t.Fatal("timeout batch did not drain")
			}
			runtime.ProcessNextEvent()
			if runtime.Clock >= 6 && runtime.Clock < 32 {
				cancelObserved = true
				if cache.GetRequestCachedPrefix(b).Tokens != 0 || len(cache.batchPrefixes.submitted) != 1 {
					t.Fatal("cancel leaked readiness or discarded submitted producer")
				}
			}
		}
		if !cancelObserved || a.State != sim.StateTimedOut || a.TTFTSet || cancelAll && (b.State != sim.StateTimedOut || b.TTFTSet) || !cancelAll && b.State != sim.StateCompleted {
			t.Fatal("wrong cancellation/output state", a.State, b.State)
		}
		for _, e := range h.records {
			if e.Name == "batch_prefix_published" && e.Time < 32 {
				t.Fatal("published before batch completion", e)
			}
		}
		if len(cache.batchPrefixes.submitted) > 0 || cache.FreeBlockCnt != cache.TotalCapacity() {
			t.Fatal("submitted donation lease did not drain")
		}
		assertPeerConservation(t, cache)
	}
}

func batchPrefixFixture(t *testing.T) (*peerHarness, *PeerCache, *sim.Request, *sim.Request) {
	h, s, _ := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2})
	if err := s.EnableBatchPrefixReuse(); err != nil {
		t.Fatal(err)
	}
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{101, 102, 103, 104, 105}, OutputTokens: []sim.TokenID{9}, State: sim.StateQueued}
	b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{101, 102, 103, 104, 999}, OutputTokens: []sim.TokenID{9}, State: sim.StateQueued}
	return h, s, a, b
}

func formPrefixPair(s *PeerCache, a, b *sim.Request, limits map[string]int64) (sim.BatchResult, map[string]int64) {
	q := &sim.WaitQueue{}
	q.Enqueue(a)
	q.Enqueue(b)
	computed := map[string]int64{}
	r := sim.NewBatchFormation("").FormBatch(sim.BatchContext{WaitQ: q, KVCache: s, MaxNumBatchedTokens: 4, MaxNumSeqs: 2, PrefillTokenThreshold: 2, ComputedTokens: computed, TokenLimits: limits})
	return r, computed
}

func TestBatchPrefixScopeAndSubmittedCancellation(t *testing.T) {
	h, s, a, b := batchPrefixFixture(t)
	r, computed := formPrefixPair(s, a, b, nil)
	if len(r.RunningBatch.Requests) != 2 || computed[a.ID] != 2 || computed[b.ID] != 4 {
		t.Fatal("same-batch hit did not reduce query", computed)
	}
	bid := s.RequestMap[a.ID][0]
	if s.RequestMap[b.ID][0] != bid || s.Blocks[bid].RefCount != 3 {
		t.Fatal("missing shared ownership/submitted lease")
	}
	if got := s.GetRequestCachedPrefix(b); got.Tokens != 0 {
		t.Fatal("scope leaked into global lookup", got)
	}
	state := s.DecisionState([]sim.DecisionKVQuery{{ID: "outside", Input: b.InputTokens}})
	if state.Requests[0].LocalPrefixTokens != 0 {
		t.Fatal("unexecuted producer appeared ready to policy")
	}
	// Cancelling both outputs cannot release storage still written by the batch.
	a.State = sim.StateTimedOut
	b.State = sim.StateTimedOut
	s.ReleaseKVBlocks(a)
	s.ReleaseKVBlocks(b)
	if s.Blocks[bid].RefCount != 1 {
		t.Fatal("cancel released in-flight donated content")
	}
	s.clock = 32
	s.CompletePrefixBatch()
	if s.FreeBlockCnt != s.TotalCapacity() || s.GetRequestCachedPrefix(b).Tokens != 2 {
		t.Fatal("completion failed to publish/release submitted content")
	}
	var dependencies, published int
	for _, e := range h.records {
		if e.Name == "batch_prefix_dependency" {
			dependencies++
		}
		if e.Name == "batch_prefix_published" {
			published++
			if e.Time != 32 {
				t.Fatal("early publication")
			}
		}
	}
	if dependencies != 1 || published != 1 {
		t.Fatal(dependencies, published)
	}
	assertPeerConservation(t, s)
}

func TestBatchPrefixOffersRespectActualQuotaAndFailedAdmission(t *testing.T) {
	_, s, a, b := batchPrefixFixture(t)
	_, computed := formPrefixPair(s, a, b, map[string]int64{"a": 1})
	if computed[b.ID] != 2 || len(s.batchPrefixes.submitted) != 0 {
		t.Fatal("partial uncompleted block was reused", computed)
	}
	_, s, a, b = batchPrefixFixture(t)
	s.KVCacheState = NewKVCacheState(3, 2)
	b.InputTokens = append(b.InputTokens, 1000, 1001)
	r, computed := formPrefixPair(s, a, b, nil)
	if len(r.RunningBatch.Requests) != 1 || computed[b.ID] != 0 || len(s.batchPrefixes.submitted) != 0 {
		t.Fatal("failed consumer admission acquired donation", computed)
	}
	if len(s.RequestMap[b.ID]) != 0 || len(s.holds[b.ID]) != 0 {
		t.Fatal("failed admission leaked ownership")
	}
}

func TestBatchPrefixRejectsMissingOrTruncatedFinalProducer(t *testing.T) {
	for _, removed := range []bool{false, true} {
		_, s, a, b := batchPrefixFixture(t)
		s.BeginPrefixBatch()
		if !s.AllocateKVBlocks(a, 0, 2, nil) {
			t.Fatal("producer allocation")
		}
		a.State = sim.StateRunning
		a.NumNewTokens = 2
		s.OfferPrefixWork(sim.BatchWork{Request: a, NewTokens: 2})
		prefix := s.GetRequestCachedPrefix(b)
		if prefix.Tokens != 2 || !s.AllocateKVBlocks(b, 2, 4, prefix.Blocks) {
			t.Fatal("consumer allocation")
		}
		work := []sim.BatchWork{{Request: a, NewTokens: 1}}
		if removed {
			work = nil
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Error("invalid donor submitted")
				}
			}()
			s.CommitPrefixBatch(work)
		}()
		if len(s.batchPrefixes.submitted) != 0 {
			t.Fatal("rejected commit acquired submission lease")
		}
		s.AbortPrefixBatch()
		s.ReleaseKVBlocks(a)
		s.ReleaseKVBlocks(b)
		assertPeerConservation(t, s)
	}
}

func TestBatchPrefixRemovedOfferAndPartialBlockCompletion(t *testing.T) {
	_, s, a, b := batchPrefixFixture(t)
	s.BeginPrefixBatch()
	if !s.AllocateKVBlocks(a, 0, 2, nil) {
		t.Fatal("allocation")
	}
	a.State = sim.StateRunning
	a.NumNewTokens = 2
	s.OfferPrefixWork(sim.BatchWork{Request: a, NewTokens: 2})
	a.NumNewTokens = 0
	s.ReleaseKVBlocks(a)
	if s.GetRequestCachedPrefix(b).Tokens != 0 {
		t.Fatal("removed producer remained visible")
	}
	s.AbortPrefixBatch()
	_, s, a, b = batchPrefixFixture(t)
	if !s.AllocateKVBlocks(a, 0, 1, nil) {
		t.Fatal("partial seed")
	}
	a.ProgressIndex = 1
	a.State = sim.StateRunning
	q := &sim.WaitQueue{}
	q.Enqueue(b)
	computed := map[string]int64{"a": 1}
	sim.NewBatchFormation("").FormBatch(sim.BatchContext{RunningBatch: &sim.Batch{Requests: []*sim.Request{a}}, WaitQ: q, KVCache: s, MaxNumBatchedTokens: 4, MaxNumSeqs: 2, PrefillTokenThreshold: 2, ComputedTokens: computed, TokenLimits: map[string]int64{"a": 1}})
	if computed[b.ID] != 4 || len(s.batchPrefixes.submitted) != 1 {
		t.Fatal("newly completed partial block unavailable", computed)
	}
	s.CompletePrefixBatch()
	s.ReleaseKVBlocks(a)
	s.ReleaseKVBlocks(b)
	assertPeerConservation(t, s)
}
