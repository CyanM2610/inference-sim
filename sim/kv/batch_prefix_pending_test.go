package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"math"
	"testing"
)

func TestBatchPrefixRuntimeRejectsMissingCompletionEvents(t *testing.T) {
	_, cache, a, _ := batchPrefixFixture(t)
	cfg := sim.SimConfig{Horizon: math.MaxInt64, KVCacheConfig: sim.NewKVCacheConfig(6, 2, 0, 0, 0, 0), BatchConfig: sim.NewBatchConfig(2, 4, 2), Seed: 42}
	runtime, err := sim.NewSimulator(cfg, cache, decisionServiceLatency{})
	if err != nil {
		t.Fatal(err)
	}
	cache.BindEvents(runtime.Schedule, runtime.ScheduleStepIfIdle)
	runtime.InjectArrival(a)
	defer func() {
		if recover() == nil {
			t.Error("batch reuse ran without its completion boundary")
		}
	}()
	runtime.Run()
}

func TestBatchPrefixCancelledProducerPublishesForPendingRestoreConsumer(t *testing.T) {
	h, cache, a, b := batchPrefixFixture(t)
	// Initial CPU state contains only the consumer's second block. Its first
	// block will come from the producer in this batch; no full CPU prefix hit.
	b.InputTokens = []sim.TokenID{101, 102, 201, 202, 203}
	keys := cache.fullPrefixHashes(b.InputTokens)
	cache.fabric.pools["cxl"].entries[keys[1]] = &peerEntry{hash: keys[1], tokens: append([]sim.TokenID(nil), b.InputTokens[2:4]...), ready: true}
	cfg := sim.SimConfig{Horizon: math.MaxInt64, KVCacheConfig: sim.NewKVCacheConfig(6, 2, 0, 0, 0, 0), BatchConfig: sim.NewBatchConfig(2, 4, 2), BatchCompletionEvents: true, Seed: 42}
	runtime, err := sim.NewSimulator(cfg, cache, decisionServiceLatency{})
	if err != nil {
		t.Fatal(err)
	}
	cache.BindEvents(runtime.Schedule, runtime.ScheduleStepIfIdle)
	a.Deadline = 5
	runtime.InjectArrival(a)
	runtime.InjectArrival(b)
	var sawHeld, restored bool
	for n := 0; runtime.HasPendingEvents(); n++ {
		if n > 1000 {
			t.Fatal("pending restore consumer failed to drain")
		}
		runtime.ProcessNextEvent()
		if a.State == sim.StateTimedOut && runtime.Clock < 32 {
			sawHeld = len(cache.holds[b.ID]) > 0 && cache.readPending[b.ID] > 0
			if len(cache.batchPrefixes.submitted) != 1 || cache.GetRequestCachedPrefix(b).Tokens != 0 {
				t.Fatal("uncompleted donation exposed/lost")
			}
		}
	}
	for _, e := range h.records {
		if e.Name == "transfer_adopted" && e.Request == b.ID && e.Destination == "hbm" {
			restored = true
		}
	}
	if !sawHeld || !restored || a.TTFTSet || a.State != sim.StateTimedOut || b.State != sim.StateCompleted {
		t.Fatal("mixed donation/restore cancellation path missing", sawHeld, restored, a.State, b.State)
	}
	if cache.FreeBlockCnt != cache.TotalCapacity() || len(cache.batchPrefixes.submitted) > 0 || len(cache.holds) > 0 {
		t.Fatal("pending consumer leaked ownership")
	}
	assertPeerConservation(t, cache)
}
