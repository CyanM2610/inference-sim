package sim

import "testing"

type asyncBatchStore struct {
	KVStore
	accept bool
	work   []BatchWork
	done   func(int64)
}
type disabledAsyncStore struct{ asyncBatchStore }

func (*disabledAsyncStore) AsyncBatchEnabled() bool { return false }
func TestDisabledAsyncBackendDoesNotInspectLegacyTokenAccounting(t *testing.T) {
	k := &disabledAsyncStore{}
	r := &Request{ID: "legacy", InputTokens: make([]TokenID, 16), NumNewTokens: 1}
	s := &Simulator{KVCache: k, RunningBatch: &Batch{Requests: []*Request{r}}}
	if s.beginAsyncBatch(0) || k.work != nil {
		t.Fatal("disabled backend touched legacy batch")
	}
}

func (k *asyncBatchStore) BeginBatch(now int64, w []BatchWork, done func(int64)) (bool, error) {
	k.work = w
	k.done = done
	return k.accept, nil
}

func TestAsyncBatchUsesActualCompletionAndPrefix(t *testing.T) {
	k := &asyncBatchStore{accept: true}
	r := &Request{ID: "prefix_hit", InputTokens: make([]TokenID, 80), NumNewTokens: 16}
	s := &Simulator{KVCache: k, RunningBatch: &Batch{Requests: []*Request{r}}, reqNumComputedTokens: map[string]int64{r.ID: 80}}
	if !s.beginAsyncBatch(100) {
		t.Fatal("backend not selected")
	}
	if len(k.work) != 1 || k.work[0].PrefixTokens != 64 || k.work[0].NewTokens != 16 {
		t.Fatal("first-admission prefix lost")
	}
	if s.HasPendingEvents() || s.stepEvent == nil {
		t.Fatal("completion was predicted before backend completion")
	}
	k.done(137)
	if !s.HasPendingEvents() || s.PeekNextEventTime() != 137 {
		t.Fatal("actual completion did not schedule batch publication")
	}
	event, ok := s.eventQueue[0].event.(*BatchCompleteEvent)
	if !ok || event.Duration != 37 || event.Start != 100 {
		t.Fatal("wrong completion envelope")
	}
	defer func() {
		if recover() == nil {
			t.Error("duplicate callback accepted")
		}
	}()
	k.done(140)
}
func TestAsyncBatchDeclinePreservesLegacyMarker(t *testing.T) {
	k := &asyncBatchStore{}
	r := &Request{ID: "decode", InputTokens: make([]TokenID, 16), ProgressIndex: 20, NumNewTokens: 1}
	previous := &StepEvent{}
	s := &Simulator{KVCache: k, RunningBatch: &Batch{Requests: []*Request{r}}, stepEvent: previous}
	if s.beginAsyncBatch(100) || s.stepEvent != previous || s.HasPendingEvents() {
		t.Fatal("decline changed legacy scheduling")
	}
	if k.work[0].PrefixTokens != 20 {
		t.Fatal("decode context changed")
	}
}
