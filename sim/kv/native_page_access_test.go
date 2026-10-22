package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type nativePageAccessCost struct{ nativePhaseTestCost }

func (nativePageAccessCost) ExecutionResources() []string {
	return []string{"cpu", "compute", "copy", "mp"}
}
func (nativePageAccessCost) TransferPlan(sim.NativeTransferShape) sim.NativeTransferPlan {
	return sim.NativeTransferPlan{Stages: []sim.NativeStage{
		{Name: "submit", Resource: "mp", Stream: "mp_host", DurationUS: 1},
		{Name: "model", DurationUS: 2000, Delay: true, After: []string{"submit"}},
		{Name: "copy", Resource: "copy", Stream: "native-kv-stream", DurationUS: 1000, After: []string{"submit"}},
		{Name: "reply", Resource: "mp", Stream: "mp_reply", DurationUS: 1, After: []string{"copy", "model"}},
	}, ProducerWait: "copy", GPUComplete: "copy", ModelComplete: "model", Published: "reply"}
}

// Native007's allocation sequence: B has 15 tokens, A has 16, two usable
// 16-token pages. Step 2 preempts A; consume step 1 releases its allocation;
// step 3 reserves that page for B while the old A STORE is still pending.
// The allocator may reuse it, but B's actual access must wait for the STORE.
func TestNativePageReuseReservesBeforeStoreButComputesAfterCompletion(t *testing.T) {
	var records []PeerRecord
	e, s, f := nativePageEngine(t, 2, func(r PeerRecord) { records = append(records, r) })
	checkNativePageReuse(t, e, s, f, &records)
}

func nativePageEngine(t *testing.T, blocks int64, record func(PeerRecord)) (*sim.Simulator, *PeerCache, *PeerFabric) {
	t.Helper()
	f, err := NewPeerFabric([]PeerResource{{ID: "link", BytesPerUS: 1}},
		[]PeerPoolConfig{{ID: "dram", CapacityBlocks: 8}}, 100,
		record)
	if err != nil {
		t.Fatal(err)
	}
	if err = f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "dram", GroupTransfers: true, ConcurrentRestores: true, RestoreWindow: 64}); err != nil {
		t.Fatal(err)
	}
	s, err := NewPeerCache("native", blocks, 16, f, []PeerAccess{{Pool: "dram", ReadPath: []string{"link"}, WritePath: []string{"link"}}}, BuiltinPeerPolicy{Name: "lru_drop"})
	if err != nil {
		t.Fatal(err)
	}
	if err = s.ConfigureHostChunks(1, OrderedHostPlacement{PoolOrder: []string{"dram"}, EvictPools: []string{"dram"}}); err != nil {
		t.Fatal(err)
	}
	if err = s.ConfigureNativeHostGeometry(1, 64<<20); err != nil {
		t.Fatal(err)
	}
	e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
		KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: blocks, BlockSizeTokens: 16},
		BatchConfig:   sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 32}}, s, nativeTestCost{})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.EnableNativeAsync(2, nativePageAccessCost{}); err != nil {
		t.Fatal(err)
	}
	return e, s, f
}

func checkNativePageReuse(t *testing.T, e *sim.Simulator, s *PeerCache, f *PeerFabric, records *[]PeerRecord) {
	t.Helper()
	b := &sim.Request{ID: "otherB", InputTokens: make([]sim.TokenID, 15), OutputTokens: []sim.TokenID{7, 8, 9, 10}, MaxOutputLen: 4}
	a := &sim.Request{ID: "oldA", InputTokens: make([]sim.TokenID, 16), OutputTokens: []sim.TokenID{17, 18}, MaxOutputLen: 2}
	for i := range a.InputTokens {
		a.InputTokens[i] = 1
	}
	oldPage, reservedAt, accessedAt := int64(-1), int64(-1), int64(-1)
	e.EventObserver = func(event sim.Event, before bool) {
		x, ok := event.(*sim.NativeAsyncEvent)
		if !ok || before {
			return
		}
		if x.Name == "native_async_scheduled" && x.BatchID == 1 {
			oldPage = s.RequestMap[a.ID][0]
		}
		if x.Name == "native_async_scheduled" && x.BatchID == 3 {
			ids := s.RequestMap[b.ID]
			if len(ids) == 2 && ids[1] == oldPage {
				reservedAt = x.At
			}
		}
		if x.Name == "native_phase_start" && x.BatchID == 3 && x.Phase == "forward" {
			accessedAt = x.At
		}
	}
	e.InjectArrival(b)
	e.InjectArrival(a)
	e.Run()
	storeEnd := int64(-1)
	for _, r := range *records {
		if r.Name == "transfer_end" && r.Request == a.ID && r.Source == "hbm" && storeEnd < 0 {
			storeEnd = r.Time
		}
	}
	if !(oldPage >= 0 && reservedAt >= 0 && reservedAt < storeEnd && accessedAt >= storeEnd) {
		t.Fatalf("native reserve/access boundary diverged: page=%d reserve=%d storeEnd=%d access=%d", oldPage, reservedAt, storeEnd, accessedAt)
	}
	// Additional simulator liveness check: native007 itself aborts after step 3.
	if e.Metrics.CompletedRequests != 2 || e.Metrics.TotalOutputTokens != 6 || s.UsedBlocks() != 0 || f.Pending() != 0 || len(s.nativePageAccess) != 0 {
		t.Fatal("full continuation did not complete and drain", s.PeerSnapshot())
	}
}

func TestNativePageReadOnlyPrefixDoesNotWaitForPendingStore(t *testing.T) {
	storeEnd, forward := int64(-1), int64(-1)
	e, s, f := nativePageEngine(t, 4, func(r PeerRecord) {
		if r.Name == "transfer_end" && r.Source == "hbm" && storeEnd < 0 {
			storeEnd = r.Time
		}
	})
	r := &sim.Request{ID: "decode", InputTokens: make([]sim.TokenID, 16), OutputTokens: []sim.TokenID{1, 2, 3, 4}, MaxOutputLen: 4}
	e.EventObserver = func(event sim.Event, before bool) {
		if x, ok := event.(*sim.NativeAsyncEvent); ok && !before && x.BatchID == 2 {
			if x.Name == "native_phase_start" && x.Phase == "forward" {
				forward = x.At
			}
			if x.Name == "native_execution_plan" && len(x.KVAccess.Dependencies) != 0 {
				t.Fatal("immutable read acquired a write hazard")
			}
		}
	}
	e.InjectArrival(r)
	e.Run()
	if !(forward > 0 && forward < storeEnd) {
		t.Fatal("read/read access serialized", forward, storeEnd)
	}
	if e.Metrics.CompletedRequests != 1 || f.Pending() != 0 || s.UsedBlocks() != 0 || len(s.nativePageAccess) != 0 {
		t.Fatal("read control did not drain", s.PeerSnapshot())
	}
}

func TestNativePageLoadConflictsWithComputeReadAndConnectorOnlyWrite(t *testing.T) {
	s := &PeerCache{KVCacheState: NewKVCacheState(2, 16), nativePageAccess: map[int64]nativePendingPageAccess{
		1: {ids: []int64{0}, write: true, fence: &sim.NativeFence{}},
		2: {ids: []int64{1}, write: false, fence: &sim.NativeFence{}},
	}}
	b := &nativeCapturedBatch{store: s, works: []nativeCapturedWork{{ids: []int64{0, 1}, work: sim.NativeWork{Start: 16, End: 17}}}}
	p := b.KVAccessPlan()
	if len(p.Dependencies) != 2 || !p.Dependencies[0].PendingWrite || p.Dependencies[1].PendingWrite {
		t.Fatal("LOAD read / STORE write conflicts missing", p)
	}
	b.works = nil
	b.loads = []*peerJob{{record: PeerRecord{HBMBlocks: []int64{0, 1}}}}
	if p = b.KVAccessPlan(); len(p.Dependencies) != 2 || len(p.ReadBlocks) != 0 {
		t.Fatal("connector-only LOAD overwrote a pending access", p)
	}
}
