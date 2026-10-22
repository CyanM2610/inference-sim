package kv

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func hostChunkHarness(t *testing.T, dram, cxl int64) (*peerHarness, *PeerCache) {
	t.Helper()
	h := &peerHarness{}
	var err error
	h.f, err = NewPeerFabric([]PeerResource{{ID: "dram", BytesPerUS: 10, LatencyUS: 2}, {ID: "cxl", BytesPerUS: 5, LatencyUS: 3}, {ID: "pcie", BytesPerUS: 20}},
		[]PeerPoolConfig{{ID: "dram", CapacityBlocks: dram}, {ID: "cxl", CapacityBlocks: cxl}}, 100, func(r PeerRecord) { h.records = append(h.records, r) })
	if err != nil {
		t.Fatal(err)
	}
	if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "dram", GroupTransfers: true, ConcurrentRestores: true, RestoreWindow: 64}); err != nil {
		t.Fatal(err)
	}
	s, err := NewPeerCache("native", 24, 2, h.f, []PeerAccess{
		{Pool: "dram", ReadPath: []string{"dram", "pcie"}, WritePath: []string{"dram", "pcie"}},
		{Pool: "cxl", ReadPath: []string{"cxl", "pcie"}, WritePath: []string{"cxl", "pcie"}},
	}, BuiltinPeerPolicy{Name: "lru_drop"})
	if err != nil {
		t.Fatal(err)
	}
	s.BindEvents(func(e sim.Event) { h.events = append(h.events, e) }, func(int64) {})
	if err := s.ConfigureHostChunks(2, OrderedHostPlacement{PoolOrder: []string{"dram", "cxl"}, EvictPools: []string{"cxl"}}); err != nil {
		t.Fatal(err)
	}
	return h, s
}

func drainHost(t *testing.T, h *peerHarness, s *PeerCache) {
	t.Helper()
	for len(h.events) > 0 {
		h.next(t)
		assertPeerConservation(t, s)
	}
}

func TestHostChunksAtomicMixedPoolPublicationAndRequestHold(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	r := seedPeer(t, s, "writer", []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9})
	if len(h.f.hostChunks.objects) != 2 || h.f.Snapshot()["dram"]["reserved"] != 2 || h.f.Snapshot()["cxl"]["reserved"] != 2 {
		t.Fatal("wrong whole-object reservation")
	}
	if s.UsedBlocks() != 5 {
		t.Fatal("native request STORE hold must include the unstored partial tail", s.UsedBlocks())
	}
	h.next(t)
	if h.f.Snapshot()["dram"]["ready"] != 0 || s.UsedBlocks() != 5 {
		t.Fatal("published or released before complete mixed-pool operation")
	}
	drainHost(t, h, s)
	if s.UsedBlocks() != 0 || h.f.Snapshot()["dram"]["ready"] != 2 || h.f.Snapshot()["cxl"]["ready"] != 2 {
		t.Fatal("publication/drain failed")
	}
	before := len(h.records)
	s.MirrorToCPU([]*sim.Request{r})
	if len(h.records) != before || h.f.Pending() != 0 {
		t.Fatal("completed request resubmitted its chunks")
	}
}

func TestHostChunksDivergentSuffixHasNoFalsePrefixHit(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	a := []sim.TokenID{1, 2, 3, 4, 5}
	b := []sim.TokenID{1, 2, 9, 9, 5}
	miss := []sim.TokenID{1, 2, 8, 8, 5}
	seedPeer(t, s, "a", a)
	drainHost(t, h, s)
	seedPeer(t, s, "b", b)
	drainHost(t, h, s)
	key := s.hashes(a)[0]
	ea, aa := s.findForInput("lookup-a", key, a)
	eb, ab := s.findForInput("lookup-b", key, b)
	if ea == nil || eb == nil || ea == eb || aa.Pool != "dram" || ab.Pool != "cxl" {
		t.Fatal("different host objects sharing a GPU prefix were deduplicated")
	}
	if e, _ := s.findForInput("miss", key, miss); e != nil {
		t.Fatal("partial host chunk falsely counted as a lookup hit")
	}
	if len(h.f.pools["dram"].entries)+len(h.f.pools["cxl"].entries) != 4 {
		t.Fatal("lost duplicate physical prefix bytes")
	}
}

func TestHostChunkPartialRestorePaysWholeObjectAndPinsIt(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	if err := s.ConfigureNativeHostGeometry(1, 64<<20); err != nil {
		t.Fatal(err)
	}
	input := []sim.TokenID{1, 2, 3, 4, 5}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	ids := s.GetCachedBlocks(input)
	s.invalidate(s.Blocks[ids[1]])
	q := &sim.Request{ID: "reader", InputTokens: input}
	if s.AllocateKVBlocks(q, 0, 5, s.GetCachedBlocks(input)) {
		t.Fatal("admitted before LOAD")
	}
	var load *PeerRecord
	for i := range h.records {
		r := &h.records[i]
		if r.Name == "transfer_start" && r.Destination == "hbm" {
			load = r
		}
	}
	if load == nil || load.Bytes != 200 || load.EffectiveBytes != 100 || len(load.HostObjects) != 1 || load.HostChunkBlocks != 2 {
		t.Fatal("partial suffix did not pay whole chunk", load)
	}
	if len(load.NativeObjects) != 1 || load.NativeObjects[0].StagingBytes != 200 || load.NativeObjects[0].KernelBytes != 100 ||
		load.NativeObjects[0].SkipTokens != 2 || len(load.NativeObjects[0].HBMBlocks) != 1 {
		t.Fatal("incorrect native suffix geometry", load.NativeObjects)
	}
	for _, record := range h.records {
		if record.Name == "transfer_end" && record.Source == "hbm" {
			for _, object := range record.NativeObjects {
				if len(object.HBMBlocks) != 2 {
					t.Fatal("STORE lost captured source geometry", object)
				}
			}
		}
	}
	for _, object := range h.f.hostChunks.objects {
		if object.evictable() {
			t.Fatal("read pin did not protect complete object")
		}
	}
	drainHost(t, h, s)
	if !s.AllocateKVBlocks(q, 2, 5, s.GetCachedBlocks(input)) {
		t.Fatal("restore never became usable")
	}
	s.ReleaseKVBlocks(q)
	assertPeerConservation(t, s)
	if s.UsedBlocks() != 0 || h.f.Snapshot()["dram"]["read_pins"] != 0 {
		t.Fatal("restore ownership leaked")
	}
}

func TestNativeHostGeometryFirstFitSlotsAndRegistrationSegments(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 10)
	h.f.bytes = 7 << 20 // two HBM blocks form the real 14 MiB dense host object
	if err := s.ConfigureNativeHostGeometry(4096, 64<<20); err != nil {
		t.Fatal(err)
	}
	var copies []sim.NativeTransferObject
	for i := 0; i < 7; i++ {
		base := sim.TokenID(10 * i)
		seedPeer(t, s, fmt.Sprintf("slot-%d", i), []sim.TokenID{base + 1, base + 2, base + 3, base + 4, base + 5})
		drainHost(t, h, s)
	}
	for _, r := range h.records {
		if r.Name == "transfer_start" {
			copies = append(copies, r.NativeObjects...)
		}
	}
	if len(copies) != 7 {
		t.Fatal("missing host geometry", len(copies))
	}
	boundary, reused := copies[5], copies[6]
	if boundary.Pool != "cxl" || boundary.HostOffset != 56<<20 || len(boundary.Segments) != 2 ||
		boundary.Segments[0].Bytes != 8<<20 || boundary.Segments[1].Bytes != 6<<20 || boundary.Segments[1].HostOffset != 64<<20 {
		t.Fatal("registered-region boundary was not preserved", boundary)
	}
	if reused.Pool != "cxl" || reused.HostOffset != 0 {
		t.Fatal("first-fit did not reuse the released lowest slot", reused)
	}
}

func TestHostChunkCancellationKeepsSubmittedPinsUntilCompletion(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, id := range s.GetCachedBlocks(input) {
		s.invalidate(s.Blocks[id])
	}
	q := &sim.Request{ID: "cancelled", InputTokens: input}
	if s.AllocateKVBlocks(q, 0, 5, nil) {
		t.Fatal("LOAD was not deferred")
	}
	s.ClearDeferred(q.ID)
	if h.f.Snapshot()["dram"]["read_pins"] != 2 {
		t.Fatal("cancel released submitted source")
	}
	drainHost(t, h, s)
	if h.f.Snapshot()["dram"]["read_pins"] != 0 || s.UsedBlocks() != 0 || len(s.holds[q.ID]) != 0 {
		t.Fatal("cancelled transfer failed to drain")
	}
}

func TestHostChunkPromotionUsesSameFullObjectAccounting(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, id := range s.GetCachedBlocks(input) {
		s.invalidate(s.Blocks[id])
	}
	if s.Promote(input, 1) != 1 {
		t.Fatal("promotion did not reserve its explicit target")
	}
	var load *PeerRecord
	for i := range h.records {
		r := &h.records[i]
		if r.Name == "transfer_start" && r.Reason == "promotion" {
			load = r
		}
	}
	if load == nil || load.Bytes != 200 || load.EffectiveBytes != 100 {
		t.Fatal("promotion got a cheaper copy mechanism", load)
	}
	drainHost(t, h, s)
	if len(s.GetCachedBlocks(input)) != 1 || s.UsedBlocks() != 0 {
		t.Fatal("promotion ownership/publication failed")
	}
}

func TestHostChunkConcurrentReadersKeepIndependentNativeTargets(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, id := range s.GetCachedBlocks(input) {
		s.invalidate(s.Blocks[id])
	}
	a := &sim.Request{ID: "reader-a", InputTokens: input}
	b := &sim.Request{ID: "reader-b", InputTokens: input}
	for _, q := range []*sim.Request{a, b} {
		if s.AllocateKVBlocks(q, 0, 5, nil) {
			t.Fatal("read admitted before native LOAD")
		}
	}
	if s.readPending[a.ID] != 2 || s.readPending[b.ID] != 2 || h.f.Snapshot()["dram"]["read_pins"] != 4 {
		t.Fatal("concurrent native readers were merged into one physical restore")
	}
	drainHost(t, h, s)
	pa, pb := s.GetRequestCachedPrefix(a), s.GetRequestCachedPrefix(b)
	if len(pa.Blocks) != 2 || len(pb.Blocks) != 2 || pa.Blocks[0] == pb.Blocks[0] {
		t.Fatal("request lost its own restored targets", pa, pb)
	}
	for _, q := range []*sim.Request{a, b} {
		p := s.GetRequestCachedPrefix(q)
		if !s.AllocateKVBlocks(q, p.Tokens, 5, p.Blocks) {
			t.Fatal("completed reader not admitted")
		}
	}
	if s.RequestMap[a.ID][0] == s.RequestMap[b.ID][0] {
		t.Fatal("adoption silently replaced an independent target")
	}
	for _, q := range []*sim.Request{a, b} {
		s.ReleaseKVBlocks(q)
	}
	assertPeerConservation(t, s)
	if s.UsedBlocks() != 0 || h.f.Snapshot()["dram"]["read_pins"] != 0 {
		t.Fatal("shared source refs did not drain")
	}
}

func TestHostChunkUnpublishedObjectIsAMissNotLegacyHitPending(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	input := []sim.TokenID{1, 2, 3, 4, 5}
	seedPeer(t, s, "writer", input)
	// Exercise lookup control independently of existing local HBM hits.
	s.restoreControl = true
	s.ensureHostLookup("lookup", input)
	known, ready, deferred := s.observeRestoreLookup(&sim.Request{ID: "lookup", InputTokens: input}, nil)
	if !known || ready != 0 || deferred {
		t.Fatal("LMCache unpublished object inherited legacy HIT_PENDING semantics")
	}
	drainHost(t, h, s)
}

func TestHostChunkImpossibleAdmissionKeepsVictims(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	seedPeer(t, s, "fill", []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9})
	drainHost(t, h, s)
	before := h.f.Snapshot()
	old := map[string]*hostObject{}
	for key, object := range h.f.hostChunks.objects {
		old[key] = object
	}
	// Only the CXL object is reclaimable by this policy, but two are needed.
	seedPeer(t, s, "impossible", []sim.TokenID{11, 12, 13, 14, 15, 16, 17, 18, 19})
	if !reflect.DeepEqual(before, h.f.Snapshot()) || !reflect.DeepEqual(old, h.f.hostChunks.objects) || h.f.Pending() != 0 {
		t.Fatal("failed admission changed retained KV")
	}
	// One-object replacement remains legal and evicts the entire CXL object.
	seedPeer(t, s, "replacement", []sim.TokenID{21, 22, 23, 24, 25})
	drainHost(t, h, s)
	for key, object := range old {
		if object.pool == "dram" && h.f.hostChunks.objects[key] != object {
			t.Fatal("native baseline evicted DRAM")
		}
		if object.pool == "cxl" && h.f.hostChunks.objects[key] != nil {
			t.Fatal("LRU CXL object not replaced")
		}
	}
}

type hostPlacementFunc func(HostPlacementView) HostPlacementPlan

func (f hostPlacementFunc) Place(v HostPlacementView) HostPlacementPlan { return f(v) }

func TestHostChunkInvalidPolicyHasNoDestructiveSideEffects(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 2)
	seedPeer(t, s, "fill", []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9})
	drainHost(t, h, s)
	before := h.f.Snapshot()
	h.f.hostChunks.policy = hostPlacementFunc(func(v HostPlacementView) HostPlacementPlan {
		v.Pools["dram"] = HostPoolSpace{Capacity: 9999}
		v.Incoming[0] = "policy-mutated"
		return HostPlacementPlan{Destinations: []string{"dram"}, Victims: []string{v.Candidates[0].Key}}
	})
	// Force a plan whose claimed DRAM capacity cannot cover its destination.
	for _, object := range h.f.hostChunks.objects {
		if object.pool == "dram" {
			object.entries[0].readers++
		}
	}
	defer func() {
		if recover() == nil {
			t.Fatal("invalid host policy accepted")
		}
		after := h.f.Snapshot()
		before["dram"]["read_pins"] = 1
		if !reflect.DeepEqual(before, after) {
			t.Fatal("invalid plan mutated pools", before, after)
		}
	}()
	seedPeer(t, s, "incoming", []sim.TokenID{31, 32, 33, 34, 35})
}
