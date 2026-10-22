package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

type nativeTestCost struct{}

func (nativeTestCost) ScheduleUS(int, int) int64             { return 2 }
func (nativeTestCost) WorkerUS([]sim.NativeBatchShape) int64 { return 100 }
func (nativeTestCost) OutputUS([]sim.NativeBatchShape) int64 { return 25 }
func (nativeTestCost) ConsumeUS(int) int64                   { return 3 }
func (nativeTestCost) StepTime([]*sim.Request) int64         { return 100 }
func (nativeTestCost) QueueingTime(*sim.Request) int64       { return 0 }
func (nativeTestCost) OutputTokenProcessingTime() int64      { return 0 }
func (nativeTestCost) PostDecodeFixedOverhead() int64        { return 0 }

func nativeAsyncHarness(t *testing.T, tokens int64, capacity ...int64) (*sim.Simulator, *PeerCache, *peerHarness) {
	t.Helper()
	h, store := hostChunkHarness(t, 2, 4)
	blocks := int64(24)
	if len(capacity) > 0 {
		blocks = capacity[0]
		store.KVCacheState = NewKVCacheState(blocks, 2)
	}
	engine, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
		KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: blocks, BlockSizeTokens: 2},
		BatchConfig:   sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: tokens}}, store, nativeTestCost{})
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.EnableNativeAsync(2, nativeTestCost{}); err != nil {
		t.Fatal(err)
	}
	return engine, store, h
}

// Same public ownership sequence as pinned Scheduler._free_request_blocks:
// request a last ran in step 1, but retirement is fenced by global step 2.
// A newer allocation with the same request ID must not extend that old fence.
func TestNativeAsyncRetirementUsesGlobalConsumedStep(t *testing.T) {
	for _, resume := range []bool{false, true} {
		t.Run(map[bool]string{false: "no_resume", true: "new_generation"}[resume], func(t *testing.T) {
			_, s, _ := nativeAsyncHarness(t, 4)
			a := &sim.Request{ID: "a", State: sim.StateRunning, InputTokens: []sim.TokenID{1}, MaxOutputLen: 2}
			b := &sim.Request{ID: "b", State: sim.StateRunning, InputTokens: []sim.TokenID{2}, MaxOutputLen: 2}
			capture := func(r *sim.Request, generation uint64) sim.NativeBatchKV {
				if !s.ReserveNativeRange(r, 0, 1, nil) {
					t.Fatal("fixture capacity unexpectedly exhausted")
				}
				return s.CaptureNativeBatch([]sim.NativeWork{{Request: r, Generation: generation, End: 1, KnownTokens: 1}})
			}
			consume := func(batch sim.NativeBatchKV, at int64) {
				batch.DeviceComplete(at)
				batch.BeforeConsume(at)
				batch.Consumed(at)
			}
			old := capture(a, 0)
			empty := s.CaptureNativeBatch(nil)
			oldID := s.RequestMap[a.ID][0]
			other := capture(b, 0)
			s.ReleaseKVBlocks(a)
			var newer sim.NativeBatchKV
			newID := int64(-1)
			if resume {
				newer = capture(a, 1)
				newID = s.RequestMap[a.ID][0]
				if newID == oldID {
					t.Fatal("retired page was reused before its fence")
				}
			}
			consume(old, 10)
			if s.Blocks[oldID].RefCount != 1 {
				t.Fatalf("step 1 must retain a until global fence 2: refs=%d", s.Blocks[oldID].RefCount)
			}
			// Empty connector polls must not satisfy a computing-step fence.
			consume(empty, 11)
			if s.Blocks[oldID].RefCount != 1 {
				t.Fatal("empty poll advanced a deferred-free fence")
			}
			consume(other, 20)
			if s.Blocks[oldID].RefCount != 0 {
				t.Fatalf("global step 2 must release the old allocation even if a resumed: refs=%d", s.Blocks[oldID].RefCount)
			}
			if resume {
				if s.Blocks[newID].RefCount == 0 || s.RequestMap[a.ID][0] != newID {
					t.Fatal("old retirement released the current generation")
				}
				consume(newer, 30)
				s.ReleaseKVBlocks(a)
			}
			s.ReleaseKVBlocks(b)
			if s.UsedBlocks() != 0 || s.PeerSnapshot()["native_deferred_free_refs"] != 0 {
				t.Fatal("retirement fixture did not drain", s.PeerSnapshot())
			}
		})
	}
}

func TestNativeAsyncRetirementPreservesNativeFreeQueueOrder(t *testing.T) {
	_, s, _ := nativeAsyncHarness(t, 4)
	// Full known blocks exercise the cached (tail/FIFO) branch. Native partial
	// blocks have no hash and return at the head; that is a separate contract.
	a := &sim.Request{ID: "a", State: sim.StateRunning, InputTokens: []sim.TokenID{1, 2}, MaxOutputLen: 1}
	b := &sim.Request{ID: "b", State: sim.StateRunning, InputTokens: []sim.TokenID{3, 4}, MaxOutputLen: 1}
	for _, r := range []*sim.Request{a, b} {
		if !s.ReserveNativeRange(r, 0, 2, nil) {
			t.Fatal("fixture reservation failed")
		}
	}
	aID, bID := s.RequestMap[a.ID][0], s.RequestMap[b.ID][0]
	batch := s.CaptureNativeBatch([]sim.NativeWork{
		{Request: a, End: 2, KnownTokens: 2}, {Request: b, End: 2, KnownTokens: 2},
	})
	// Native preemption retires newest running requests first. Capture iteration
	// order is A,B and must not override the retired allocation order B,A.
	s.ReleaseKVBlocks(b)
	s.ReleaseKVBlocks(a)
	batch.DeviceComplete(10)
	batch.BeforeConsume(10)
	if s.Blocks[aID].RefCount != 0 || s.Blocks[bID].RefCount != 0 {
		t.Fatal("capture ownership outlived the native consumption fence")
	}
	if s.FreeTail.ID != aID || s.FreeTail.PrevFree.ID != bID {
		t.Fatal("capture order changed native LRU free order")
	}
	batch.Consumed(10)
	if s.FreeTail.ID != aID || s.FreeTail.PrevFree.ID != bID || s.UsedBlocks() != 0 {
		t.Fatal("output bookkeeping changed retired page order")
	}
}

func TestNativeAsyncFreeQueuePrefersUncachedPages(t *testing.T) {
	for _, retired := range []bool{false, true} {
		t.Run(map[bool]string{false: "immediate", true: "retired"}[retired], func(t *testing.T) {
			_, s, _ := nativeAsyncHarness(t, 6, 6)
			s.fabric.hostChunks.policy = hostPlacementFunc(func(HostPlacementView) HostPlacementPlan { return HostPlacementPlan{} })
			r := &sim.Request{ID: "tail", State: sim.StateRunning, InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, MaxOutputLen: 1}
			if !s.ReserveNativeRange(r, 0, 5, nil) {
				t.Fatal("fixture allocation failed")
			}
			ids := append([]int64(nil), s.RequestMap[r.ID]...)
			if s.Blocks[ids[0]].Hash == "" || s.Blocks[ids[1]].Hash == "" || s.Blocks[ids[2]].Hash != "" {
				t.Fatal("fixture does not separate cached and partial pages")
			}
			freeBefore := s.FreeHead.ID
			var batch sim.NativeBatchKV
			if retired {
				batch = s.CaptureNativeBatch([]sim.NativeWork{{Request: r, End: 5, KnownTokens: 5}})
			}
			s.ReleaseKVBlocks(r)
			if retired {
				batch.DeviceComplete(10)
				batch.BeforeConsume(10)
				batch.Consumed(10)
			}
			if s.FreeHead.ID != ids[2] || s.FreeHead.NextFree.ID != freeBefore || s.FreeTail.ID != ids[0] || s.FreeTail.PrevFree.ID != ids[1] {
				t.Fatalf("native must reuse the uncached tail before cached/older free pages: head=%d tail=%d", s.FreeHead.ID, s.FreeTail.ID)
			}
			if s.UsedBlocks() != 0 || s.FreeBlockCnt != 6 {
				t.Fatal("free queue conservation failed")
			}
		})
	}
}

// Same 6-block, two 5+5-token requests as the native CPU scheduler control.
// No STORE/LOAD occurs in either control. Native kv_both repeatedly preempts
// both requests before delivering their second sample; do not hide this with
// a scheduler wait. The test bounds observation rather than requiring success.
func TestNativeAsyncPressurePreservesNativeNonConvergence(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 10, 6)
	s.fabric.hostChunks.policy = hostPlacementFunc(func(HostPlacementView) HostPlacementPlan { return HostPlacementPlan{} })
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6, 7, 8, 9, 10}, MaxOutputLen: 5}
	b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{11, 12, 13, 14, 15}, OutputTokens: []sim.TokenID{16, 17, 18, 19, 20}, MaxOutputLen: 5}
	consumed := int64(0)
	e.EventObserver = func(event sim.Event, before bool) {
		if x, ok := event.(*sim.NativeAsyncEvent); ok && !before && x.Name == "native_async_consumed" {
			consumed = x.BatchID
		}
	}
	e.InjectArrival(a)
	e.InjectArrival(b)
	for i := 0; e.HasPendingEvents() && consumed < 127 && i < 10000; i++ {
		e.ProcessNextEvent()
	}
	if consumed != 127 || e.Metrics.CompletedRequests != 0 || a.EmittedTokens() != 1 || b.EmittedTokens() != 1 || e.Metrics.PreemptionCount != 126 {
		t.Fatalf("native 128-step pressure control diverged: consumed=%d completed=%d outputs=%d/%d preemptions=%d", consumed, e.Metrics.CompletedRequests, a.EmittedTokens(), b.EmittedTokens(), e.Metrics.PreemptionCount)
	}
	if s.fabric.Pending() != 0 || len(s.fabric.hostChunks.objects) != 0 {
		t.Fatal("no-transfer control unexpectedly used host retention")
	}
}

func TestNativeAsyncPreemptionCompletesAfterShortPeerFinishes(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 10, 6)
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6, 7}, MaxOutputLen: 2}
	b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{11, 12, 13, 14, 15}, OutputTokens: []sim.TokenID{16, 17, 18, 19, 20}, MaxOutputLen: 5}
	e.InjectArrival(a)
	e.InjectArrival(b)
	e.Run()
	if e.Metrics.PreemptionCount == 0 || e.Metrics.CompletedRequests != 2 || e.Metrics.TotalOutputTokens != 7 || a.EmittedTokens() != 2 || b.EmittedTokens() != 5 {
		t.Fatal("preemption lost/duplicated observed output", e.Metrics.PreemptionCount, a, b)
	}
	if len(a.ITL) != 1 || len(b.ITL) != 4 || s.UsedBlocks() != 0 || s.nativeBatchPins != 0 || len(s.nativePending) != 0 {
		t.Fatal("preemption did not finish cleanly", s.PeerSnapshot())
	}
}

// Positive controls from the same native CPU probe. No host STORE/LOAD in
// either implementation; timings and token identities are not compared.
func TestNativeAsyncNativePressureCompletionControls(t *testing.T) {
	for _, c := range []struct {
		name        string
		blocks      int64
		firstBudget int
		steps       int64
		preemptions int64
	}{{"sufficient_capacity", 10, 5, 7, 0}, {"short_peer", 6, 2, 9, 1}} {
		t.Run(c.name, func(t *testing.T) {
			e, s, _ := nativeAsyncHarness(t, 10, c.blocks)
			s.fabric.hostChunks.policy = hostPlacementFunc(func(HostPlacementView) HostPlacementPlan { return HostPlacementPlan{} })
			a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6, 7, 8, 9, 10}[:c.firstBudget], MaxOutputLen: c.firstBudget}
			b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{11, 12, 13, 14, 15}, OutputTokens: []sim.TokenID{16, 17, 18, 19, 20}, MaxOutputLen: 5}
			last := int64(0)
			finished := map[string]int{}
			e.EventObserver = func(event sim.Event, before bool) {
				if x, ok := event.(*sim.NativeAsyncEvent); ok && !before {
					if x.Name == "native_async_consumed" {
						last = x.BatchID
					}
					if x.Name == "native_async_scheduled" {
						for _, id := range x.FinishedRequests {
							finished[id]++
						}
					}
				}
			}
			e.InjectArrival(a)
			e.InjectArrival(b)
			e.Run()
			if last != c.steps || e.Metrics.PreemptionCount != c.preemptions || a.EmittedTokens() != int64(c.firstBudget) || b.EmittedTokens() != 5 || e.Metrics.CompletedRequests != 2 {
				t.Fatalf("native completion control diverged: last=%d preemptions=%d outputs=%d/%d completed=%d", last, e.Metrics.PreemptionCount, a.EmittedTokens(), b.EmittedTokens(), e.Metrics.CompletedRequests)
			}
			if s.UsedBlocks() != 0 || s.nativeBatchPins != 0 || len(s.nativePending) != 0 || len(s.nativeDeferred) != 0 || s.fabric.Pending() != 0 {
				t.Fatal("control did not drain")
			}
			if len(finished) != 2 || finished[a.ID] != 1 || finished[b.ID] != 1 {
				t.Fatal("finished metadata missing or duplicated", finished)
			}
		})
	}
}

func TestNativeAsyncLMCacheProducerDoesNotDeliverPreemptedSamples(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 10, 6)
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6, 7}, MaxOutputLen: 2}
	b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{11, 12, 13, 14, 15}, OutputTokens: []sim.TokenID{16, 17, 18, 19, 20}, MaxOutputLen: 5}
	byID := map[string]*sim.Request{"a": a, "b": b}
	beforeOutput := map[string]int64{}
	checked := 0
	e.EventObserver = func(event sim.Event, before bool) {
		x, ok := event.(*sim.NativeAsyncEvent)
		if !ok || x.Name != "native_async_consumed" {
			return
		}
		if before {
			beforeOutput = map[string]int64{}
			for _, shape := range x.Shape {
				r := byID[shape.Request]
				if r.State == sim.StateQueued {
					beforeOutput[r.ID] = r.EmittedTokens()
				}
			}
		} else {
			for id, count := range beforeOutput {
				checked++
				if byID[id].EmittedTokens() != count {
					t.Fatal("LMCache kv_both delivered a stale preempted sample", id, count, byID[id].EmittedTokens())
				}
			}
		}
	}
	e.InjectArrival(a)
	e.InjectArrival(b)
	e.Run()
	if checked == 0 || e.Metrics.CompletedRequests != 2 || e.Metrics.TotalOutputTokens != 7 || s.nativeBatchPins != 0 || s.UsedBlocks() != 0 {
		t.Fatal("preemption scenario was not exercised or failed to drain", checked, s.PeerSnapshot())
	}
}

func TestNativeAsyncNativeAPCAndClosedLoopArrival(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 16)
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}, OutputTokens: []sim.TokenID{10}, MaxOutputLen: 1}
	b := &sim.Request{ID: "b", InputTokens: append([]sim.TokenID(nil), a.InputTokens...), OutputTokens: []sim.TokenID{10}, MaxOutputLen: 1}
	child := &sim.Request{ID: "child", InputTokens: []sim.TokenID{31, 32, 33}, OutputTokens: []sim.TokenID{34}, MaxOutputLen: 1}
	first := true
	parentAt := int64(-1)
	e.EventObserver = func(event sim.Event, before bool) {
		if x, ok := event.(*sim.NativeAsyncEvent); ok && !before && x.Name == "native_async_scheduled" && first {
			first = false
			if len(x.Shape) != 2 || x.Shape[1].ContextTokens != 8 || x.Shape[1].QueryTokens != 1 {
				t.Fatal("native allocate_slots APC registration missing", x.Shape)
			}
		}
	}
	e.OnRequestDone = func(r *sim.Request, at int64) []*sim.Request {
		if r == b {
			parentAt = at
			child.ArrivalTime = at + 17
			return []*sim.Request{child}
		}
		return nil
	}
	e.InjectArrival(a)
	e.InjectArrival(b)
	e.Run()
	if first || parentAt < 0 || child.ArrivalTime != parentAt+17 || e.Metrics.CompletedRequests != 3 || !child.TTFTSet {
		t.Fatal("closed-loop child was not driven by actual consumption")
	}
	if s.UsedBlocks() != 0 || e.Metrics.PeakKVBlocksUsed == 0 || e.Metrics.KVBlocksUsed == 0 {
		t.Fatal("native metrics/drain missing")
	}
}

func TestNativeAsyncSchedulesAheadAndPreservesSnapshots(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 2)
	r := &sim.Request{ID: "one", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6, 7, 8}, MaxOutputLen: 3}
	var events []*sim.NativeAsyncEvent
	e.EventObserver = func(event sim.Event, before bool) {
		if !before {
			if x, ok := event.(*sim.NativeAsyncEvent); ok {
				events = append(events, x)
			}
		}
	}
	e.InjectArrival(r)
	e.Run()
	if r.State != sim.StateCompleted || e.Metrics.CompletedRequests != 1 || e.Metrics.TotalOutputTokens != 3 || len(r.ITL) != 2 {
		t.Fatal("request did not complete exactly", r, e.Metrics.CompletedRequests)
	}
	var scheduled, consumed []*sim.NativeAsyncEvent
	for _, x := range events {
		if x.Name == "native_async_scheduled" {
			scheduled = append(scheduled, x)
		}
		if x.Name == "native_async_consumed" {
			consumed = append(consumed, x)
		}
	}
	if len(scheduled) < 2 || scheduled[1].At >= consumed[0].At {
		t.Fatal("did not schedule before previous host consumption")
	}
	if len(scheduled[0].Shape) != 1 || scheduled[0].Shape[0].ContextTokens != 0 || scheduled[0].Shape[0].QueryTokens != 2 || scheduled[1].Shape[0].ContextTokens != 2 {
		t.Fatal("mutable batch work corrupted snapshots", scheduled)
	}
	if s.UsedBlocks() != 0 || s.nativeBatchPins != 0 || len(s.nativeDeferred) != 0 || s.fabric.Pending() != 0 {
		t.Fatal("physical ownership did not drain", s.PeerSnapshot())
	}
}

func TestNativeAsyncCancellationKeepsSubmittedPagesUntilConsumption(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 4)
	r := &sim.Request{ID: "cancel", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}, OutputTokens: []sim.TokenID{10, 11}, MaxOutputLen: 2, Deadline: 10}
	checked := false
	e.EventObserver = func(event sim.Event, before bool) {
		if !before && event.Timestamp() == 10 && r.State == sim.StateTimedOut {
			checked = true
			if s.UsedBlocks() == 0 || s.nativeBatchPins == 0 || len(s.nativeDeferred) == 0 {
				t.Fatal("cancellation freed submitted targets")
			}
		}
	}
	e.InjectArrival(r)
	e.Run()
	if !checked || r.TTFTSet || e.Metrics.TimedOutRequests != 1 || e.Metrics.CompletedRequests != 0 {
		t.Fatal("cancel produced an output", r)
	}
	if s.UsedBlocks() != 0 || s.nativeBatchPins != 0 || s.fabric.Pending() != 0 || len(s.nativeDeferred) != 0 {
		t.Fatal("cancelled work did not drain", s.PeerSnapshot())
	}
}

func TestNativeAsyncEarlyEOSDoesNotEmitQueuedExtraOutput(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 16)
	r := &sim.Request{ID: "eos", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6}, MaxOutputLen: 8}
	e.InjectArrival(r)
	e.Run()
	if e.Metrics.TotalOutputTokens != 1 || len(r.ITL) != 0 || r.EmittedTokens() != 1 || e.Metrics.CompletedRequests != 1 {
		t.Fatal("queued output survived EOS", r)
	}
	if s.UsedBlocks() != 0 || s.nativeBatchPins != 0 {
		t.Fatal("EOS pins leaked")
	}
}

func TestNativeAsyncRejectedStoreIsNotRetriedByLaterDecode(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 16)
	calls := 0
	s.fabric.hostChunks.policy = hostPlacementFunc(func(v HostPlacementView) HostPlacementPlan {
		calls++
		if calls == 1 {
			return HostPlacementPlan{} // server admission failure after metadata
		}
		return OrderedHostPlacement{PoolOrder: []string{"dram", "cxl"}, EvictPools: []string{"cxl"}}.Place(v)
	})
	r := &sim.Request{ID: "rejected-store", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6, 7}, MaxOutputLen: 2}
	e.InjectArrival(r)
	e.Run()
	if e.Metrics.CompletedRequests != 1 || calls != 1 || len(s.fabric.hostChunks.objects) != 0 {
		t.Fatal("later decode implicitly retried an already offered STORE", calls, s.PeerSnapshot())
	}
	if len(s.nativeStored) != 0 || s.UsedBlocks() != 0 || s.fabric.Pending() != 0 {
		t.Fatal("STORE metadata or physical ownership leaked")
	}
}

func TestNativeAsyncIncrementalStoreCanEvictEarlierPrefixChunk(t *testing.T) {
	e, s, _ := nativeAsyncHarness(t, 4)
	// One host object total: each later STORE must be allowed to replace the
	// earlier, unlocked chunk of the same request. Protect only current keys.
	s.fabric.pools["dram"].config.CapacityBlocks = 0
	s.fabric.pools["cxl"].config.CapacityBlocks = 2
	r := &sim.Request{ID: "incremental", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}, OutputTokens: []sim.TokenID{10}, MaxOutputLen: 1}
	e.InjectArrival(r)
	e.Run()
	last := s.hashes(r.InputTokens)[3]
	if e.Metrics.CompletedRequests != 1 || len(s.fabric.hostChunks.objects) != 1 || s.fabric.hostChunks.objects[last] == nil {
		t.Fatal("incremental STORE unnecessarily protected the earlier chunk", s.fabric.hostChunks.objects)
	}
	if len(s.nativeStored) != 0 || s.UsedBlocks() != 0 || s.fabric.Pending() != 0 {
		t.Fatal("incremental STORE did not drain")
	}
}

func TestNativeAsyncLoadIsPublishedOnlyAtHostConsumption(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	input := []sim.TokenID{1, 2, 3, 4}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, block := range s.Blocks {
		if block.Hash != "" {
			s.invalidate(block)
		}
	}
	e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
		KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 24, BlockSizeTokens: 2}, BatchConfig: sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 16}}, s, nativeTestCost{})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.EnableNativeAsync(2, nativeTestCost{}); err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "read", InputTokens: input, OutputTokens: []sim.TokenID{5}, MaxOutputLen: 1}
	checked := false
	firstCompute := int64(-1)
	e.EventObserver = func(event sim.Event, before bool) {
		if !before && s.readPending[r.ID] == 0 && len(s.nativeLoadTargets[r.ID]) > 0 {
			checked = true
			if len(s.GetCachedBlocks(append(input, 9))) != 0 {
				t.Fatal("DMA completion exposed a prefix before native adoption")
			}
		}
		if x, ok := event.(*sim.NativeAsyncEvent); ok && !before && x.Name == "native_async_scheduled" && len(x.Shape) > 0 && firstCompute < 0 {
			firstCompute = x.At
			if x.Shape[0].ContextTokens != 3 || x.Shape[0].QueryTokens != 1 {
				t.Fatal("full host hit failed one-token logits replay", x.Shape)
			}
		}
	}
	e.InjectArrival(r)
	e.Run()
	adopted := int64(-1)
	for _, record := range h.records {
		if record.Name == "native_receive_adopted" && record.Request == r.ID {
			adopted = record.Time
		}
	}
	if !checked || adopted < 0 || firstCompute <= adopted || !r.TTFTSet || e.Metrics.CompletedRequests != 1 {
		t.Fatal("missing native receive boundary", adopted, firstCompute, checked)
	}
	if s.UsedBlocks() != 0 || len(s.nativeLoadPending) != 0 || s.fabric.Pending() != 0 {
		t.Fatal("restored request did not drain", s.PeerSnapshot())
	}
}
