package sim

import "testing"

type readyAdmissionStore struct {
	KVStore
	ready []string
}

func (s *readyAdmissionStore) ReadyDeferredRequests() []string { return s.ready }

func TestDeferredAdmissionPreservesPolicyOrderWithinReadyGroup(t *testing.T) {
	store := &readyAdmissionStore{KVStore: MustNewKVCacheState(10, 2), ready: []string{"first", "second"}}
	queue := &WaitQueue{}
	requests := []*Request{
		{ID: "fresh", InputTokens: []TokenID{1, 2, 3}},
		{ID: "second", InputTokens: []TokenID{4, 5, 6}},
		{ID: "first", InputTokens: []TokenID{7, 8, 9}},
	}
	for _, r := range requests {
		queue.Enqueue(r)
	}
	result := NewBatchFormation("fcfs").FormBatch(BatchContext{
		WaitQ: queue, KVCache: store, MaxNumBatchedTokens: 16, MaxNumSeqs: 2,
		ComputedTokens: map[string]int64{},
	})
	if len(result.NewlyScheduled) != 2 || result.NewlyScheduled[0].Request.ID != "second" || result.NewlyScheduled[1].Request.ID != "first" || queue.Peek().ID != "fresh" {
		t.Fatal("ready reservations must precede fresh requests without overriding the request policy's order")
	}
}
