package sim

import (
	"reflect"
	"testing"
)

type historyTestStore struct{ KVStore }

func (historyTestStore) PreservesOutputHistory() bool { return true }

func TestDecodePreemptionKeepsOutputAndRecomputesObservedHistory(t *testing.T) {
	s, _ := serviceFixture(t, nil)
	s.KVCache = historyTestStore{s.KVCache}
	r := &Request{ID: "decoded", InputTokens: []TokenID{1, 2, 3, 4}, OutputTokens: []TokenID{90, 91, 92, 93, 94},
		State: StateRunning, ProgressIndex: 5, TTFTSet: true, FirstTokenTime: 10, ITL: []int64{10}}
	ctx := BatchContext{KVCache: s.KVCache, ComputedTokens: s.reqNumComputedTokens}
	resetPreemptedRequest(r, ctx)
	if r.ProgressIndex != 0 || !r.TTFTSet || r.FirstTokenTime != 10 || !reflect.DeepEqual(r.ITL, []int64{10}) {
		t.Fatal("KV eviction erased observed output or TTFT", r)
	}
	s.WaitQ.PrependFront(r)
	s.RunningBatch = &Batch{}
	for i := 0; i < 3; i++ {
		v := s.decisionView(int64(100 + i*10))
		var visible DecisionRequest
		if len(v.Waiting) > 0 {
			visible = v.Waiting[0]
		} else {
			visible = v.Running[0]
		}
		if visible.InputTokens != 4 || visible.EmittedTokens != 2 {
			t.Fatal("recompute exposed a different prompt/output", visible)
		}
		ctx := BatchContext{KVCache: s.KVCache, RunningBatch: s.RunningBatch, WaitQ: s.WaitQ, ComputedTokens: s.reqNumComputedTokens,
			MaxNumBatchedTokens: 2, MaxNumSeqs: 1, PrefillTokenThreshold: 2}
		batch := (&VLLMBatchFormation{}).FormBatch(ctx)
		s.RunningBatch = batch.RunningBatch
		if r.NumNewTokens != 2 {
			t.Fatal("history chunk was not issued", i, r.NumNewTokens)
		}
		s.executeBatchDuration(int64(100+i*10), 10)
		if r.FirstTokenTime != 10 || !r.TTFTSet {
			t.Fatal("recompute overwrote first output")
		}
		if i < 2 && len(r.ITL) != 1 {
			t.Fatal("partial history replay emitted another token", r.ITL)
		}
	}
	if r.ProgressIndex != 6 || len(r.ITL) != 2 || r.ITL[1] != 110 {
		t.Fatal("replay did not emit the next token after the complete observed history", r.ProgressIndex, r.ITL)
	}
	if got := s.decisionView(130).Running[0]; got.EmittedTokens != 3 || got.InputTokens != 4 {
		t.Fatal("new output/prompt accounting", got)
	}
	s.KVCache.ReleaseKVBlocks(r)
}

func TestRepeatedPreemptionRetainsOnlyAlreadyObservedTokens(t *testing.T) {
	s, _ := serviceFixture(t, nil)
	store := historyTestStore{s.KVCache}
	r := &Request{ID: "repeated", InputTokens: []TokenID{1, 2, 3, 4}, OutputTokens: []TokenID{90, 91, 92, 93, 94},
		ProgressIndex: 6, TTFTSet: true, FirstTokenTime: 10, ITL: []int64{10, 10}}
	ctx := BatchContext{KVCache: store, ComputedTokens: map[string]int64{}}
	resetPreemptedRequest(r, ctx)
	want := []TokenID{1, 2, 3, 4, 90, 91, 92}
	if !reflect.DeepEqual(r.PrefillTokens(), want) || r.EmittedTokens() != 3 || r.PrefillEnd() != 7 {
		t.Fatal("incorrect observed history after first preemption")
	}
	r.OutputTokens[3], r.OutputTokens[4] = 1000, 1001
	for _, partial := range []int64{2, 5} {
		r.ProgressIndex = partial
		resetPreemptedRequest(r, ctx)
		if r.EmittedTokens() != 3 || !reflect.DeepEqual(r.PrefillTokens(), want) || r.PrefillEnd() != 7 || len(r.ITL) != 2 {
			t.Fatal("partial replay lost observed history or exposed future output")
		}
	}
	if !reflect.DeepEqual(r.FullInputTokens(), []TokenID{1, 2, 3, 4}) || r.InputLen() != 4 {
		t.Fatal("recompute changed the user's prompt")
	}
}
