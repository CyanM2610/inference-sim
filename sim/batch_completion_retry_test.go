package sim

import "testing"

// A competing capacity reservation denies one running allocation. Releasing
// the running request makes its next admission feasible, without an I/O event.
type preemptionRetryStore struct {
	KVStore
	rejectRunningOnce bool
}

func (k *preemptionRetryStore) AllocateKVBlocks(r *Request, start, end int64, cached []int64) bool {
	if k.rejectRunningOnce && r.ProgressIndex > 0 {
		k.rejectRunningOnce = false
		return false
	}
	return k.KVStore.AllocateKVBlocks(r, start, end, cached)
}

func TestBatchCompletionModeRetriesAfterLastRunningRequestPreempted(t *testing.T) {
	cfg := SimConfig{KVCacheConfig: NewKVCacheConfig(20, 2, 0, 0, 0, 0), BatchConfig: NewBatchConfig(1, 16, 0), BatchCompletionEvents: true, Seed: 42}
	k := &preemptionRetryStore{KVStore: MustNewKVStoreFromConfig(cfg.KVCacheConfig)}
	s, err := NewSimulator(cfg, k, &fixedStepModel{stepTime: 10})
	if err != nil {
		t.Fatal(err)
	}
	r := &Request{ID: "running", InputTokens: []TokenID{1, 2, 3}, OutputTokens: []TokenID{8, 9}, State: StateRunning}
	if !k.AllocateKVBlocks(r, 0, 3, nil) {
		t.Fatal("initial allocation")
	}
	r.ProgressIndex = 3
	s.RunningBatch = &Batch{Requests: []*Request{r}}
	s.reqNumComputedTokens[r.ID] = 3
	k.rejectRunningOnce = true
	s.Step(100)
	if s.Metrics.PreemptionCount != 1 || !s.HasPendingEvents() {
		t.Fatal("preemption stranded waiting request")
	}
	if _, ok := s.ProcessNextEvent().(*StepEvent); !ok {
		t.Fatal("missing next scheduling attempt")
	}
	if _, ok := s.stepEvent.(*BatchCompleteEvent); !ok || len(s.RunningBatch.Requests) != 1 {
		t.Fatal("retry did not produce real work")
	}
}
