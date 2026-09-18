package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// PeerCache wraps BLIS's physical HBM allocator. DRAM and CXL are peers in
// access, not stages in a chain. Only completed full input blocks are reusable.
type PeerCache struct {
	cacheMetrics         *cacheMetricState
	hotprefix            *hotPrefixRuntime
	prefixCopies         map[string][]int64
	decodeTargets        map[string]int64
	allocationFailure    sim.AllocationFailure
	promotionRetention   string
	promotionControl     bool
	batchPrefixes        *batchPrefixState
	restoreControl       bool
	restoreLimits        map[string]int64
	restoredPrefixBlocks map[string]int64
	*KVCacheState
	id                string
	fabric            *PeerFabric
	access            []PeerAccess
	policy            PeerPolicy
	clock             int64
	schedule          func(sim.Event)
	wake              func(int64)
	decisionEvents    func(sim.DecisionEvent)
	ready             map[int64]string
	lookup            map[string]int64
	frequency         map[string]int64
	observed          map[string]bool
	waiting           map[string]bool
	restoring         map[string]bool
	holds             map[string][]int64
	readPending       map[string]int
	restoreOwner      string
	prefillTargets    map[string]int64
	cancelledRestores map[string]bool
	directory         map[string]bool
	spaceWait         map[string]bool
	heat              map[string]*peerHeat
	promoted          map[int64]bool
	pendingTargets    map[int64]bool
	storePins         map[int64]int
	backgroundOffered map[string]map[string]bool
}

func NewPeerCache(id string, blocks, blockTokens int64, fabric *PeerFabric, access []PeerAccess, policy PeerPolicy) (*PeerCache, error) {
	if blocks <= 0 || blockTokens <= 0 || policy == nil {
		return nil, fmt.Errorf("positive HBM/block sizes and a policy are required")
	}
	if err := fabric.validateAccess(access); err != nil {
		return nil, err
	}
	if pool := fabric.mechanisms.BackgroundStorePool; pool != "" {
		found := false
		for _, a := range access {
			found = found || a.Pool == pool
		}
		if !found {
			return nil, fmt.Errorf("background store pool %q is inaccessible", pool)
		}
	}
	s := &PeerCache{KVCacheState: NewKVCacheState(blocks, blockTokens), id: id, fabric: fabric, access: access, policy: policy, ready: map[int64]string{}, lookup: map[string]int64{}, frequency: map[string]int64{}, observed: map[string]bool{}, waiting: map[string]bool{}, restoring: map[string]bool{}, holds: map[string][]int64{}, readPending: map[string]int{}}
	fabric.stores = append(fabric.stores, s)
	s.directory = map[string]bool{}
	s.spaceWait = map[string]bool{}
	s.promoted = map[int64]bool{}
	s.pendingTargets = map[int64]bool{}
	s.storePins = map[int64]int{}
	s.backgroundOffered = map[string]map[string]bool{}
	s.prefillTargets = map[string]int64{}
	s.cancelledRestores = map[string]bool{}
	if fabric.mechanisms.Online != nil {
		s.heat = map[string]*peerHeat{}
	}
	return s, nil
}
func (s *PeerCache) BindEvents(schedule func(sim.Event), wake func(int64)) {
	s.schedule = schedule
	s.wake = wake
}

func (s *PeerCache) BindDecisionEvents(notify func(sim.DecisionEvent)) { s.decisionEvents = notify }
func (s *PeerCache) SetClock(t int64)                                  { s.clock = t; s.expireHotPrefixShadows() }
func (s *PeerCache) emit(name, req, h, src, dst, reason string) {
	s.fabric.emit(PeerRecord{Time: s.clock, Name: name, Instance: s.id, Request: req, Hash: h, Source: src, Destination: dst, Reason: reason})
	if s.fabric.mechanisms != (PeerMechanisms{}) {
		switch name {
		case "source_pin", "source_release", "hbm_reserve", "hbm_publish", "hbm_allocate", "request_kv_release", "promotion_cancelled":
			s.fabric.emit(PeerRecord{Time: s.clock, Name: "hbm_capacity", Instance: s.id, Counters: s.PeerSnapshot()})
		}
	}
}
func (s *PeerCache) hashes(tokens []sim.TokenID) []string {
	var out []string
	prev := ""
	// The last prompt token must execute to produce logits even for a full hit.
	for i := int64(0); i+s.BlockSizeTokens < int64(len(tokens)); i += s.BlockSizeTokens {
		prev = s.hashBlock(prev, tokens[i:i+s.BlockSizeTokens])
		out = append(out, prev)
	}
	return out
}
func (s *PeerCache) GetCachedBlocks(tokens []sim.TokenID) []int64 {
	var ids []int64
	for _, h := range s.hashes(tokens) {
		id, ok := s.lookup[h]
		if !ok {
			break
		}
		ids = append(ids, id)
	}
	return ids
}
func (s *PeerCache) SnapshotCachedBlocksFn() func([]sim.TokenID) int {
	m := map[string]bool{}
	for h := range s.lookup {
		m[h] = true
	}
	return func(tokens []sim.TokenID) int {
		n := 0
		for _, h := range s.hashes(tokens) {
			if !m[h] {
				break
			}
			n++
		}
		return n
	}
}
func (s *PeerCache) pin(b *KVBlock) {
	if b.RefCount == 0 {
		s.removeFromFreeList(b)
		b.InUse = true
	}
	b.RefCount++
}
func (s *PeerCache) unpin(b *KVBlock) {
	b.RefCount--
	if b.RefCount == 0 {
		b.InUse = false
		s.appendToFreeList(b)
	}
}
func (s *PeerCache) invalidate(b *KVBlock) {
	if s.cacheMetrics != nil {
		delete(s.cacheMetrics.origins, b.ID)
	}
	if x := s.fabric.external; x != nil && x.memory != nil {
		if err := x.memory.InvalidateHBM(b.ID); err != nil {
			panic(err)
		}
	}
	if n := s.fabric.native; n != nil && n.pagedEnabled {
		n.advance(s.clock)
		if err := n.Runtime.Memory.Invalidate(n.hbmID(b.ID)); err != nil {
			panic(err)
		}
	}
	if s.promoted[b.ID] {
		s.emit("promotion_unused", "", b.Hash, "hbm", "", "evicted_before_use")
		delete(s.promoted, b.ID)
	}
	s.forgetReadyCopy(b)
	if id, ok := s.HashToBlock[b.Hash]; ok && id == b.ID {
		delete(s.HashToBlock, b.Hash)
	}
	b.Hash = ""
	b.Tokens = nil
}
func (s *PeerCache) copies(h string) []string {
	var out []string
	for _, a := range s.access {
		if e := s.fabric.pools[a.Pool].entries[h]; e != nil && e.ready {
			out = append(out, a.Pool)
		}
	}
	return out
}
func (s *PeerCache) targets() []PeerTarget {
	var free, full []PeerTarget
	for _, a := range s.access {
		p := s.fabric.pools[a.Pool]
		t := PeerTarget{Pool: a.Pool, StoreUS: s.fabric.duration(a.WritePath)}
		if s.fabric.mechanisms.QueueAware {
			t.QueueUS = s.fabric.EstimateQueueUS(s.clock, a.WritePath)
		}
		if int64(len(p.entries)) < p.config.CapacityBlocks {
			t.Available = true
			free = append(free, t)
		} else {
			for _, e := range p.entries {
				if e.ready && e.readers == 0 {
					t.Available = true
					break
				}
			}
			full = append(full, t)
		}
	}
	return append(free, full...)
}

// makeSpace prepares allocator slots. Source-reuse mode protects the old bytes
// with execution dependencies; legacy mode holds the slot until adoption.
func (s *PeerCache) makeSpace(n int64, protect map[int64]bool, req string) bool {
	var empty []*KVBlock
	var candidates []PeerCandidate
	for b := s.FreeHead; b != nil; b = b.NextFree {
		if protect[b.ID] {
			continue
		}
		if b.Hash == "" || s.ready[b.ID] != b.Hash {
			s.invalidate(b)
			empty = append(empty, b)
		} else {
			candidates = append(candidates, PeerCandidate{ID: b.ID, Hash: b.Hash, Frequency: s.frequency[b.Hash], ReadyCopies: s.copies(b.Hash)})
		}
	}
	// STORE sources leave the free list until completion is adopted. Retrying
	// after a partial completion must include that outstanding physical space,
	// rather than selecting another full set of victims on every wakeup.
	// A source adopted by a request, or needed as a protected hit, cannot pay
	// this allocation's deficit. Multiple STORE pins still promise one block.
	promised := s.reclaimingBlocks(protect)
	s.spaceWait[req] = int64(len(empty)) < n
	for int64(len(empty))+promised < n && len(candidates) > 0 {
		context := PeerReclaimContext{Candidates: candidates, Targets: s.targets()}
		if s.hotprefix != nil {
			context.ResidentHashes = s.hotPrefixResidents()
		}
		d := s.policy.Choose(cloneReclaimContext(context))
		s.annotateHotPrefixChoice(req, n-int64(len(empty))-promised)
		if d.Decline {
			s.emit("reclaim_declined", req, "", "hbm", "", "no_eligible_victim")
			break
		}
		idx := -1
		for i, c := range candidates {
			if c.ID == d.BlockID {
				idx = i
				break
			}
		}
		if idx < 0 {
			panic("peer policy selected a non-candidate block")
		}
		b := s.Blocks[d.BlockID]
		if s.hotprefix != nil {
			s.hotPrefixRecord("hotprefix_reclaim", req, b.Hash, d.Pool)
		}
		candidates = append(candidates[:idx], candidates[idx+1:]...)
		s.emit("reclaim_decision", req, b.Hash, "hbm", d.Pool, "policy")
		stored := d.Pool != "" && s.store(b, d.Pool, req)
		if stored && !s.storeSourceReuseEnabled() {
			s.waiting[req] = true
			promised++
			continue
		}
		s.emit("hbm_drop", req, b.Hash, "hbm", "", "reclaim")
		droppedHash := b.Hash
		s.invalidate(b)
		s.shadowHotPrefixDrop(droppedHash, req)
		empty = append(empty, b)
	}
	if int64(len(empty)) < n && promised > 0 {
		s.waiting[req] = true
	}
	// Move all prepared empties before cached free blocks. Do not move protected hits.
	for i := len(empty) - 1; i >= 0; i-- {
		b := empty[i]
		s.removeFromFreeList(b)
		b.PrevFree = nil
		b.NextFree = s.FreeHead
		if s.FreeHead != nil {
			s.FreeHead.PrevFree = b
		} else {
			s.FreeTail = b
		}
		s.FreeHead = b
		s.FreeBlockCnt++
	}
	if int64(len(empty)) < n {
		s.noteAllocationPressure(req, "hbm_space", n, int64(len(empty)), protect)
		return false
	}
	return true
}
func (s *PeerCache) store(b *KVBlock, pool, req string) bool {
	return s.storeCopy(b, pool, req, false)
}

func (s *PeerCache) storeCopy(b *KVBlock, pool, req string, retain bool) bool {
	var path []string
	for _, a := range s.access {
		if a.Pool == pool {
			path = a.WritePath
			break
		}
	}
	if path == nil {
		panic("peer policy selected an inaccessible pool")
	}
	h := b.Hash
	// A background observation of the same block never adds a second source
	// pin or waiter, whether the CPU copy is ready or still being written.
	if retain && s.fabric.pools[pool].entries[h] != nil {
		return false
	}
	e, fresh := s.fabric.reserve(pool, h, b.Tokens, s.clock)
	if e == nil {
		return false
	}
	if e.ready {
		return false
	} // an existing copy permits immediate local drop
	return s.storeReservedCopy(b, pool, req, retain, path, e, fresh)
}

// The caller owns an existing unready destination reservation. Request spill
// uses this after reserving the full prefix; ordinary storeCopy reserves one.
func (s *PeerCache) storeReservedCopy(b *KVBlock, pool, req string, retain bool, path []string, e *peerEntry, fresh bool) bool {
	h := b.Hash
	// Both background and reclaim STOREs protect old contents independently
	// of allocator ownership. Actual compute/load reuse installs the fence.
	leased := s.storeSourceReuseEnabled()
	if leased && !fresh {
		// The existing transaction reads its original physical source; this
		// duplicate cached copy does not acquire a fictitious content lease.
		s.emit("store_join", req, h, "hbm", pool, "deduplicated")
		return true
	}
	var sourceBlocks []int64
	if leased {
		sourceBlocks = []int64{b.ID}
		s.emit("source_lease", req, h, "hbm", pool, "store")
	} else {
		s.pin(b)
		s.storePins[b.ID]++
		s.emit("source_pin", req, h, "hbm", pool, "store")
	}
	release := func(now int64) {
		s.clock = now
		if leased {
			// b may already belong to another request/content generation.
			// Publication releases only the old lease, never that owner's refs.
			s.emit("source_lease_release", req, h, "hbm", pool, "store_complete")
			return
		}
		if !retain && b.RefCount == 1 {
			s.invalidate(b)
		}
		s.unpin(b)
		s.storePins[b.ID]--
		if s.storePins[b.ID] == 0 {
			delete(s.storePins, b.ID)
		}
		s.emit("source_release", req, h, "hbm", pool, "store_complete")
	}
	e.waiters = append(e.waiters, release)
	e.blockers = append(e.blockers, peerBlocker{request: req, active: func() bool { return s.spaceWait[req] && b.RefCount == 1 }})
	if !fresh {
		s.emit("store_join", req, h, "hbm", pool, "deduplicated")
		return true
	}
	s.fabric.submit(s.clock, &peerJob{nativeSource: func() string { return s.fabric.hbmBuffer(b.ID) }, nativeDestination: func() string { return e.nativeBuffer }, record: PeerRecord{Instance: s.id, Request: req, Source: "hbm", Destination: pool, Hash: h, HBMBlocks: sourceBlocks, Reason: "store"}, path: path, schedule: s.schedule, blockedRequests: func() []string {
		var ids []string
		for _, b := range e.blockers {
			if b.active() {
				ids = append(ids, b.request)
			}
		}
		return ids
	}, deadline: func() int64 {
		d := int64(math.MaxInt64)
		for _, blocker := range e.blockers {
			if blocker.active() {
				d = min(d, s.fabric.deadline(blocker.request))
			}
		}
		return d
	}, done: func(now int64) {
		e.ready = true
		e.last = now
		s.fabric.poolPolicyEvent(pool, "ready", req, []string{h}, now)
		s.clock = now
		s.emit("l2_publish", req, h, "hbm", pool, "store_complete")
		for _, done := range e.waiters {
			done(now)
		}
		e.waiters = nil
		e.blockers = nil
	}})
	return true
}
func (s *PeerCache) find(h string) (*peerEntry, *PeerAccess) {
	var best *PeerAccess
	var entry *peerEntry
	for i := range s.access {
		a := &s.access[i]
		e := s.fabric.pools[a.Pool].entries[h]
		cost := func(a *PeerAccess) int64 {
			d := s.fabric.duration(a.ReadPath)
			if s.fabric.mechanisms.QueueAware {
				d += s.fabric.EstimateQueueUS(s.clock, a.ReadPath)
			}
			return d
		}
		if e != nil && e.ready && (best == nil || cost(a) < cost(best)) {
			best = a
			entry = e
		}
	}
	return entry, best
}
func (s *PeerCache) fetch(h, req string, e *peerEntry, a *PeerAccess, protect map[int64]bool, background bool) bool {
	return s.fetchWithRetention(h, req, e, a, protect, background, false)
}

func (s *PeerCache) fetchWithRetention(h, req string, e *peerEntry, a *PeerAccess, protect map[int64]bool, background, retainRequest bool) bool {
	if s.restoring[h] {
		s.waiting[req] = true
		return true
	}
	if background && s.fabric.mechanisms.MaxPromotionQueueUS > 0 && s.fabric.EstimateQueueUS(s.clock, a.ReadPath) > s.fabric.mechanisms.MaxPromotionQueueUS {
		s.emit("promotion_rejected", req, h, a.Pool, "hbm", "queue_budget")
		return false
	}
	var b *KVBlock
	reserve := func(now int64) bool {
		s.clock = now
		if !s.makeSpace(1, protect, req) {
			return false
		}
		b = s.popFreeBlock()
		s.invalidate(b)
		if p := s.fabric.phases; p != nil && p.reuseSources {
			p.allocated(s.clock, req, []int64{b.ID})
		}
		b.RefCount = 1
		b.InUse = true
		s.pendingTargets[b.ID] = true
		return true
	}
	if !s.fabric.mechanisms.ReserveAtDispatch && !reserve(s.clock) {
		return false
	}
	s.restoring[h] = true
	e.readers++
	e.last = s.clock
	s.readPending[req]++
	reason := "restore"
	if background {
		reason = "promotion"
	}
	if b != nil {
		s.emit("hbm_reserve", req, h, a.Pool, "hbm", reason)
	}
	var prepare func(int64) bool
	if s.fabric.mechanisms.ReserveAtDispatch {
		prepare = func(now int64) bool {
			if !reserve(now) {
				return false
			}
			s.emit("hbm_reserve", req, h, a.Pool, "hbm", reason)
			return true
		}
	}
	var cancel func(int64)
	// Phase submissions are not cancellable and may be grouped just like
	// demand loads. Legacy queued promotion cancellation remains unchanged.
	if background && s.fabric.phases == nil {
		cancel = func(now int64) {
			s.clock = now
			e.readers--
			s.fabric.poolPolicyEvent(a.Pool, "ready", req, []string{h}, now)
			delete(s.restoring, h)
			s.readPending[req]--
			if s.readPending[req] == 0 {
				delete(s.readPending, req)
			}
			if b != nil {
				delete(s.pendingTargets, b.ID)
				s.invalidate(b)
				s.unpin(b)
			}
			s.emit("promotion_cancelled", req, h, a.Pool, "hbm", "foreground_arrival")
		}
	}
	s.fabric.submit(s.clock, &peerJob{nativeSource: func() string { return e.nativeBuffer }, nativeDestination: func() string { return s.fabric.hbmBuffer(b.ID) }, record: PeerRecord{Instance: s.id, Request: req, Source: a.Pool, Destination: "hbm", Hash: h, Reason: reason}, path: a.ReadPath, schedule: s.schedule, prepare: prepare, cancel: cancel, done: func(now int64) {
		s.clock = now
		delete(s.pendingTargets, b.ID)
		b.Hash = h
		b.Tokens = append([]sim.TokenID(nil), e.tokens...)
		s.HashToBlock[h] = b.ID
		s.publishReadyCopy(b)
		s.metricLoad(b, a.Pool, reason, req)
		s.releaseHotPrefixParent(h)
		e.readers--
		e.last = now
		s.fabric.poolPolicyEvent(a.Pool, "ready", req, []string{h}, now)
		delete(s.restoring, h)
		s.readPending[req]--
		if s.readPending[req] == 0 {
			delete(s.readPending, req)
		}
		if (background && !retainRequest) || s.cancelledRestores[req] {
			if s.fabric.mechanisms.Online != nil || s.hotprefix != nil {
				s.promoted[b.ID] = true
			}
			s.unpin(b)
		} else {
			s.holds[req] = append(s.holds[req], b.ID)
		}
		if s.readPending[req] == 0 {
			delete(s.cancelledRestores, req)
		}
		s.emit("hbm_publish", req, h, a.Pool, "hbm", reason)
	}})
	return true
}
func (s *PeerCache) AllocateKVBlocks(req *sim.Request, start, end int64, cached []int64) bool {
	s.observeCacheLookup(req.ID, req.FullInputTokens())
	s.observeHotPrefix(req)
	fresh := len(s.RequestMap[req.ID]) == 0
	s.noteAllocationWait(req.ID, "dependency")
	known, restoreCandidates, pendingLookup := s.observeRestoreLookup(req, cached)
	if known {
		defer func() {
			if s.allocationFailure.Kind != "" {
				s.allocationFailure.RestoreLookupKnown = true
				s.allocationFailure.RestoreCandidateBlocks = restoreCandidates
			}
		}()
	}
	if pendingLookup {
		return false
	}
	if !s.reserveDecodeCapacity(req, end, cached) {
		return false
	}
	if s.fabric.mechanisms.GroupTransfers {
		s.fabric.beginTransferGroup()
		defer func() { s.fabric.endTransferGroup(s.clock) }()
	}
	protect := map[int64]bool{}
	for _, id := range cached {
		protect[id] = true
	}
	if _, running := s.RequestMap[req.ID]; !running {
		if s.fabric.mechanisms.CancelQueuedPromotions {
			s.fabric.CancelQueuedPromotions(s.clock, s.id)
		}
		// Once its transfers finish, let the restore owner consume its held
		// pages before new admissions compete for the remaining capacity.
		if !s.fabric.mechanisms.ConcurrentRestores && s.restoreOwner != "" && s.restoreOwner != req.ID && s.readPending[s.restoreOwner] == 0 {
			s.waiting[req.ID] = true
			s.noteAllocationWait(req.ID, "ready_restore_owner")
			return false
		}
		hashes := s.hashes(req.FullInputTokens())
		if s.readPending[req.ID] == 0 && len(s.holds[req.ID]) == 0 {
			s.touchRequestPools(req.ID, req.FullInputTokens())
		}
		if !s.observed[req.ID] {
			for _, h := range hashes {
				s.frequency[h]++
			}
			s.observeHeat(req.FullInputTokens())
			s.observed[req.ID] = true
			s.emit("cache_lookup", req.ID, "", "", "", "request")
		}
		if s.fabric.mechanisms.ConcurrentRestores {
			if s.restoreControl {
				// A loading cap must not truncate request heat/frequency accounting.
				hashes = s.restoreHashes(req)
			}
			if !s.fullPrefillFits(req, cached) {
				// An in-flight target is already charged to HBM but is not yet
				// a usable cached hit. Wait for adoption before inferring that
				// another running request must be sacrificed for this admission.
				if s.readPending[req.ID] > 0 {
					s.noteAllocationWait(req.ID, "restore_in_flight")
				} else {
					for _, h := range hashes {
						if s.restoring[h] {
							s.noteAllocationWait(req.ID, "shared_restore_in_flight")
							break
						}
					}
				}
				return false
			}
			if s.concurrentRestore(req, cached, hashes) {
				return false
			}
		} else if len(cached) < len(hashes) {
			if len(s.access) > 0 && !s.directoryReady(req.ID) {
				return false
			}
			window := max(1, s.fabric.mechanisms.RestoreWindow)
			submitted := 0
			for _, h := range hashes[len(cached):] {
				if submitted == window {
					break
				}
				e, a := s.find(h)
				if e == nil {
					break
				}
				// Serialize foreground restores, not every new request. A
				// resident hit (or a miss handled by computation) can proceed
				// while another request owns pending restore reservations.
				if s.restoreOwner != "" && s.restoreOwner != req.ID {
					s.waiting[req.ID] = true
					return false
				}
				if !s.fetch(h, req.ID, e, a, protect, false) {
					break
				}
				// The local prefix is part of this pending admission too.
				// Pin it across other requests' allocations, otherwise they
				// can evict an early page while restored suffix pages remain
				// held, leaving a fragmented prefix that cannot make progress.
				s.holdCachedPrefix(req.ID, cached)
				submitted++
				s.restoreOwner = req.ID
				s.emit("allocation_wait", req.ID, h, a.Pool, "hbm", "restore")
			}
			if submitted > 0 {
				return false
			}
		}
	}
	newTokens := end - start
	replayLast := s.restoreControl && len(s.RequestMap[req.ID]) == 0 && start == req.PrefillEnd()-1 && end == req.PrefillEnd() &&
		int64(len(cached))*s.BlockSizeTokens == req.PrefillEnd() && s.restoredPrefixBlocks[req.ID] >= int64(len(cached))
	if req.ProgressIndex >= req.PrefillEnd() {
		newTokens = 1
	}
	if ids := s.RequestMap[req.ID]; len(ids) > 0 {
		spare := s.BlockSizeTokens - int64(len(s.Blocks[ids[len(ids)-1]].Tokens))
		newTokens -= min(spare, newTokens)
	}
	n := (newTokens + s.BlockSizeTokens - 1) / s.BlockSizeTokens
	if replayLast {
		n = 0 // logits replay writes into the already-loaded final block
	}
	if !s.makeSpace(n, protect, req.ID) {
		s.emit("allocation_wait", req.ID, "", "", "", "hbm_space")
		return false
	}
	var previous map[int64]bool
	if p := s.fabric.phases; p != nil && p.reuseSources {
		previous = map[int64]bool{}
		for _, id := range s.RequestMap[req.ID] {
			previous[id] = true
		}
		for _, id := range cached {
			previous[id] = true
		}
	}
	var ok bool
	if replayLast {
		s.commitCachedBlocks(req.ID, cached)
		ok = true
	} else {
		ok = s.KVCacheState.AllocateKVBlocks(req, start, end, cached)
	}
	if !ok {
		failure := s.KVCacheState.LastAllocationFailure()
		if failure.Kind == "capacity" {
			s.noteAllocationPressure(req.ID, failure.Reason, failure.NeededBlocks, failure.FreeBlocks, protect)
		}
	}
	if ok {
		s.stageCacheReuse(req, start, cached, fresh)
		s.stageHotPrefixReuse(req, start, cached, fresh)
		s.allocationFailure = sim.AllocationFailure{}
		if previous != nil {
			var allocated []int64
			for _, id := range s.RequestMap[req.ID] {
				if !previous[id] {
					allocated = append(allocated, id)
				}
			}
			s.fabric.phases.allocated(s.clock, req.ID, allocated)
		}
		if s.fabric.mechanisms.ConcurrentRestores {
			if end < req.PrefillEnd() {
				s.prefillTargets[req.ID] = (req.PrefillEnd() + s.BlockSizeTokens - 1) / s.BlockSizeTokens
			} else {
				delete(s.prefillTargets, req.ID)
			}
		}
		for _, id := range cached {
			if s.promoted[id] {
				s.emit("promotion_consumed", req.ID, s.Blocks[id].Hash, "hbm", "", "prefix_hit")
				delete(s.promoted, id)
			}
		}
		s.ClearDeferred(req.ID)
		s.emit("hbm_allocate", req.ID, "", "", "hbm", "admitted")
	}
	return ok
}
func (s *PeerCache) holdCachedPrefix(req string, cached []int64) {
	held := make(map[int64]bool, len(s.holds[req]))
	for _, id := range s.holds[req] {
		held[id] = true
	}
	for _, id := range cached {
		if !held[id] {
			s.pin(s.Blocks[id])
			s.holds[req] = append(s.holds[req], id)
			held[id] = true
		}
	}
}
func (s *PeerCache) MirrorToCPU(batch []*sim.Request) {
	// Compatibility method name in KVStore; peer mode publishes computed HBM
	// blocks here. An explicitly configured background pool also receives them.
	for _, req := range batch {
		s.completeCacheReuse(req)
		s.completeHotPrefixReuse(req)
		if s.hotprefix != nil {
			s.hotprefix.policy.remember(s.hotPrefixKeys(req.PrefillTokens()))
		}
		stored := false
		if s.fabric.mechanisms.GroupTransfers {
			s.fabric.beginTransferGroup()
		}
		for i, id := range s.RequestMap[req.ID] {
			b := s.Blocks[id]
			end := int64(i+1) * s.BlockSizeTokens
			if end <= min(req.ProgressIndex, req.PrefillEnd()) && b.Hash != "" && int64(len(b.Tokens)) == s.BlockSizeTokens {
				if x := s.fabric.external; x != nil && x.memory != nil && s.ready[id] != b.Hash {
					if err := x.memory.CheckContents(b.Hash, s.fabric.hbmBuffer(id)); err != nil {
						panic(err)
					}
				}
				if s.fabric.native != nil && s.ready[id] != b.Hash {
					s.fabric.native.computed(s.clock, id, b.Hash)
				}
				s.publishHotPrefixCompute(req, b)
				s.publishReadyCopy(b)
				if pool := s.fabric.mechanisms.BackgroundStorePool; pool != "" && (s.fabric.mechanisms.BackgroundStoreMode == "" || s.fabric.mechanisms.BackgroundStoreMode == "continuous") {
					if s.backgroundOffered[req.ID] == nil {
						s.backgroundOffered[req.ID] = map[string]bool{}
					}
					if !s.backgroundOffered[req.ID][b.Hash] {
						exists := s.fabric.pools[pool].entries[b.Hash] != nil
						if exists || s.storeCopy(b, pool, req.ID, true) {
							s.backgroundOffered[req.ID][b.Hash] = true
							stored = stored || !exists
						}
					}
				}
			}
		}
		if stored {
			s.touchRequestPools(req.ID, req.FullInputTokens())
		}
		if s.fabric.mechanisms.GroupTransfers {
			s.fabric.endTransferGroup(s.clock)
		}
		if h := s.hotprefix; h != nil && req.ProgressIndex >= req.PrefillEnd() && !h.prefills[req.ID] {
			h.prefills[req.ID] = true
			h.wantPromotion = h.policy.config.PromotionBlocks > 0
		}
	}
}
func (s *PeerCache) ReleaseKVBlocks(req *sim.Request) {
	if s.hotprefix != nil {
		if r := s.hotprefix.requests[req.ID]; r != nil {
			r.pending = nil
		}
	}
	if s.cacheMetrics != nil {
		if r := s.cacheMetrics.requests[req.ID]; r != nil {
			r.pending = nil
		}
	}
	s.releaseWorkerRequest(req)
	s.releaseDecodeCapacity(req.ID)
	delete(s.restoreLimits, req.ID)
	delete(s.restoredPrefixBlocks, req.ID)
	delete(s.prefillTargets, req.ID)
	s.ClearDeferred(req.ID)
	s.KVCacheState.ReleaseKVBlocks(req)
	if s.fabric.mechanisms.ReserveAtDispatch {
		s.fabric.dispatch(s.clock)
	}
	s.emit("request_kv_release", req.ID, "", "hbm", "", "finished_or_preempted")
}
func (s *PeerCache) IsDeferred(id string) bool {
	ready, started := s.directory[id]
	return s.waiting[id] || s.readPending[id] > 0 || started && !ready
}
func (s *PeerCache) PollDeferred(_ int64) []string {
	m := map[string]bool{}
	for id, v := range s.waiting {
		m[id] = v
	}
	for id, n := range s.readPending {
		m[id] = n > 0
	}
	for id, ready := range s.directory {
		if !ready {
			m[id] = true
		}
	}
	return sortedPeerIDs(m)
}
func (s *PeerCache) ClearDeferred(id string) {
	if len(s.RequestMap[id]) == 0 {
		s.releaseDecodeCapacity(id)
		delete(s.restoreLimits, id)
		delete(s.restoredPrefixBlocks, id)
	}
	if s.fabric.mechanisms.ConcurrentRestores && len(s.RequestMap[id]) == 0 {
		delete(s.prefillTargets, id)
		if s.readPending[id] > 0 {
			// DMA cannot be rolled back. Keep its target/source pins until
			// completion, then publish an unowned cache entry instead of
			// recreating a reservation for a request that has left the queue.
			s.cancelledRestores[id] = true
		}
	}
	delete(s.waiting, id)
	delete(s.directory, id)
	delete(s.spaceWait, id)
	if s.restoreOwner == id {
		s.restoreOwner = ""
		s.waiting = map[string]bool{}
	}
	for _, bid := range s.holds[id] {
		s.unpin(s.Blocks[bid])
	}
	delete(s.holds, id)
}

type peerPromotionEvent struct {
	at     int64
	store  *PeerCache
	tokens []sim.TokenID
	budget int
}

func (e *peerPromotionEvent) Timestamp() int64 { return e.at }
func (e *peerPromotionEvent) Priority() int    { return sim.PriorityAdapterLoad }
func (e *peerPromotionEvent) Execute(_ *sim.Simulator) {
	e.store.SetClock(e.at)
	e.store.Promote(e.tokens, e.budget)
}
func (s *PeerCache) SchedulePromotion(at int64, tokens []sim.TokenID, budget int) {
	s.schedule(&peerPromotionEvent{at: at, store: s, tokens: append([]sim.TokenID(nil), tokens...), budget: budget})
}

// Promote explicitly fetches a known prefix before demand. It does not inspect
// future requests. The caller supplies a time/event and a finite block budget.
func (s *PeerCache) Promote(tokens []sim.TokenID, budget int) int {
	return s.promotePrefix(tokens, budget, nil)
}
func (s *PeerCache) promotePrefix(tokens []sim.TokenID, budget int, attempted map[string]bool) int {
	count := 0
	protect := map[int64]bool{}
	for _, id := range s.GetCachedBlocks(tokens) {
		protect[id] = true
	}
	for _, h := range s.hashes(tokens) {
		if count >= budget {
			break
		}
		if _, ok := s.lookup[h]; ok {
			continue
		}
		if s.restoring[h] {
			continue
		}
		if attempted != nil {
			if attempted[h] {
				break
			}
			attempted[h] = true
		}
		e, a := s.find(h)
		if e == nil {
			break
		}
		if !s.fetch(h, "@promotion", e, a, protect, true) {
			break
		}
		count++
	}
	s.emit("promotion_decision", "@promotion", "", "", "", "explicit_budget")
	return count
}
func (s *PeerCache) PeerSnapshot() map[string]int64 {
	m := map[string]int64{"capacity": s.TotalBlocks, "free": s.FreeBlockCnt, "active_or_pinned": s.TotalBlocks - s.FreeBlockCnt, "ready": int64(len(s.ready)), "pending_reads": int64(len(s.restoring)), "held_restores": 0}
	if s.hotprefix != nil {
		m["hotprefix_history_nodes"] = int64(len(s.hotprefix.policy.nodes))
		m["hotprefix_shadow_entries"] = int64(len(s.hotprefix.shadows))
		m["hotprefix_shadow_ttl_us"] = s.hotprefix.shadowTTL
		m["hotprefix_requests_observed"] = s.hotprefix.policy.requests
		m["hotprefix_parent_pins"] = int64(len(s.hotprefix.parentHolds))
		m["promoted_unconsumed_resident"] = int64(len(s.promoted))
	}
	if p := s.fabric.phases; p != nil && p.workerResidents != nil {
		m["worker_resident_requests"] = int64(len(p.workerResidents))
	}
	for _, ids := range s.RequestMap {
		m["request_refs"] += int64(len(ids))
	}
	for _, b := range s.Blocks {
		m["total_refs"] += int64(b.RefCount)
		if len(b.Tokens) > 0 {
			m["resident"]++
		}
	}
	m["extra_refs"] = m["total_refs"] - m["request_refs"]
	if s.decodeTargets != nil {
		m["decode_reserved_blocks"] = s.decodeCapacityReserved("")
		m["decode_capacity_requests"] = int64(len(s.decodeTargets))
	}
	for _, ids := range s.holds {
		m["held_restores"] += int64(len(ids))
	}
	if s.fabric.mechanisms != (PeerMechanisms{}) {
		m["pending_hbm_targets"] = int64(len(s.pendingTargets))
		m["store_pinned_blocks"] = int64(len(s.storePins))
		if s.fabric.mechanisms.Online != nil {
			m["promoted_unconsumed_resident"] = int64(len(s.promoted))
		}
	}
	return m
}
