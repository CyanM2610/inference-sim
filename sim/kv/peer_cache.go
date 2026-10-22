package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// PeerCache wraps BLIS's physical HBM allocator. DRAM and CXL are peers in
// access, not stages in a chain. Only completed full input blocks are reusable.
type PeerCache struct {
	*KVCacheState
	id             string
	fabric         *PeerFabric
	access         []PeerAccess
	policy         PeerPolicy
	clock          int64
	schedule       func(sim.Event)
	wake           func(int64)
	ready          map[int64]string
	lookup         map[string]int64
	frequency      map[string]int64
	observed       map[string]bool
	waiting        map[string]bool
	restoring      map[string]bool
	holds          map[string][]int64
	readPending    map[string]int
	restoreOwner   string
	directory      map[string]bool
	spaceWait      map[string]bool
	heat           map[string]*peerHeat
	promoted       map[int64]bool
	pendingTargets map[int64]bool
	storePins      map[int64]int
}

func NewPeerCache(id string, blocks, blockTokens int64, fabric *PeerFabric, access []PeerAccess, policy PeerPolicy) (*PeerCache, error) {
	if blocks <= 0 || blockTokens <= 0 || policy == nil {
		return nil, fmt.Errorf("positive HBM/block sizes and a policy are required")
	}
	if err := fabric.validateAccess(access); err != nil {
		return nil, err
	}
	s := &PeerCache{KVCacheState: NewKVCacheState(blocks, blockTokens), id: id, fabric: fabric, access: access, policy: policy, ready: map[int64]string{}, lookup: map[string]int64{}, frequency: map[string]int64{}, observed: map[string]bool{}, waiting: map[string]bool{}, restoring: map[string]bool{}, holds: map[string][]int64{}, readPending: map[string]int{}}
	fabric.stores = append(fabric.stores, s)
	s.directory = map[string]bool{}
	s.spaceWait = map[string]bool{}
	s.promoted = map[int64]bool{}
	s.pendingTargets = map[int64]bool{}
	s.storePins = map[int64]int{}
	if fabric.mechanisms.Online != nil {
		s.heat = map[string]*peerHeat{}
	}
	return s, nil
}
func (s *PeerCache) BindEvents(schedule func(sim.Event), wake func(int64)) {
	s.schedule = schedule
	s.wake = wake
}
func (s *PeerCache) SetClock(t int64) { s.clock = t }
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
		prev = hash.HashBlock(prev, tokens[i:i+s.BlockSizeTokens])
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
	if s.promoted[b.ID] {
		s.emit("promotion_unused", "", b.Hash, "hbm", "", "evicted_before_use")
		delete(s.promoted, b.ID)
	}
	if id, ok := s.lookup[b.Hash]; ok && id == b.ID {
		delete(s.lookup, b.Hash)
	}
	if id, ok := s.HashToBlock[b.Hash]; ok && id == b.ID {
		delete(s.HashToBlock, b.Hash)
	}
	delete(s.ready, b.ID)
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

// makeSpace prepares empty blocks at the free-list head; the native allocator
// may then run unchanged without silently overwriting a selected STORE source.
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
	promised := int64(0)
	// STORE sources leave the free list until completion is adopted. Retrying
	// after a partial completion must include that outstanding physical space,
	// rather than selecting another full set of victims on every wakeup.
	// A source adopted by a request, or needed as a protected hit, cannot pay
	// this allocation's deficit. Multiple STORE pins still promise one block.
	for id, pins := range s.storePins {
		if pins > 0 && s.Blocks[id].RefCount == pins && !protect[id] {
			promised++
		}
	}
	s.spaceWait[req] = int64(len(empty)) < n
	for int64(len(empty))+promised < n && len(candidates) > 0 {
		d := s.policy.Choose(PeerReclaimContext{Candidates: candidates, Targets: s.targets()})
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
		candidates = append(candidates[:idx], candidates[idx+1:]...)
		s.emit("reclaim_decision", req, b.Hash, "hbm", d.Pool, "policy")
		if d.Pool != "" && s.store(b, d.Pool, req) {
			s.waiting[req] = true
			promised++
			continue
		}
		s.emit("hbm_drop", req, b.Hash, "hbm", "", "reclaim")
		s.invalidate(b)
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
	return int64(len(empty)) >= n
}
func (s *PeerCache) store(b *KVBlock, pool, req string) bool {
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
	e, fresh := s.fabric.reserve(pool, h, b.Tokens, s.clock)
	if e == nil {
		return false
	}
	if e.ready {
		return false
	} // an existing copy permits immediate local drop
	s.pin(b)
	s.storePins[b.ID]++
	s.emit("source_pin", req, h, "hbm", pool, "store")
	release := func(now int64) {
		s.clock = now
		if b.RefCount == 1 {
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
	s.fabric.submit(s.clock, &peerJob{record: PeerRecord{Instance: s.id, Request: req, Source: "hbm", Destination: pool, Hash: h, Reason: "store"}, path: path, schedule: s.schedule, blockedRequests: func() []string {
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
	if background {
		cancel = func(now int64) {
			s.clock = now
			e.readers--
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
	s.fabric.submit(s.clock, &peerJob{record: PeerRecord{Instance: s.id, Request: req, Source: a.Pool, Destination: "hbm", Hash: h, Reason: reason}, path: a.ReadPath, schedule: s.schedule, prepare: prepare, cancel: cancel, done: func(now int64) {
		s.clock = now
		delete(s.pendingTargets, b.ID)
		b.Hash = h
		b.Tokens = append([]sim.TokenID(nil), e.tokens...)
		s.HashToBlock[h] = b.ID
		s.ready[b.ID] = h
		s.lookup[h] = b.ID
		e.readers--
		e.last = now
		delete(s.restoring, h)
		s.readPending[req]--
		if s.readPending[req] == 0 {
			delete(s.readPending, req)
		}
		if background {
			if s.fabric.mechanisms.Online != nil {
				s.promoted[b.ID] = true
			}
			s.unpin(b)
		} else {
			s.holds[req] = append(s.holds[req], b.ID)
		}
		s.emit("hbm_publish", req, h, a.Pool, "hbm", reason)
	}})
	return true
}
func (s *PeerCache) AllocateKVBlocks(req *sim.Request, start, end int64, cached []int64) bool {
	protect := map[int64]bool{}
	for _, id := range cached {
		protect[id] = true
	}
	if _, running := s.RequestMap[req.ID]; !running {
		if s.fabric.mechanisms.CancelQueuedPromotions {
			s.fabric.CancelQueuedPromotions(s.clock, s.id)
		}
		// One foreground restore admission per instance bounds temporary HBM
		// holds and avoids several partially restored requests pinning each other.
		if s.restoreOwner != "" && s.restoreOwner != req.ID {
			s.waiting[req.ID] = true
			return false
		}
		hashes := s.hashes(req.FullInputTokens())
		if !s.observed[req.ID] {
			for _, h := range hashes {
				s.frequency[h]++
			}
			s.observeHeat(req.FullInputTokens())
			s.observed[req.ID] = true
			s.emit("cache_lookup", req.ID, "", "", "", "request")
		}
		if len(cached) < len(hashes) {
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
				if !s.fetch(h, req.ID, e, a, protect, false) {
					break
				}
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
	if req.ProgressIndex >= req.InputLen() {
		newTokens = 1
	}
	if ids := s.RequestMap[req.ID]; len(ids) > 0 {
		spare := s.BlockSizeTokens - int64(len(s.Blocks[ids[len(ids)-1]].Tokens))
		newTokens -= min(spare, newTokens)
	}
	n := (newTokens + s.BlockSizeTokens - 1) / s.BlockSizeTokens
	if !s.makeSpace(n, protect, req.ID) {
		s.emit("allocation_wait", req.ID, "", "", "", "hbm_space")
		return false
	}
	ok := s.KVCacheState.AllocateKVBlocks(req, start, end, cached)
	if ok {
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
func (s *PeerCache) MirrorToCPU(batch []*sim.Request) {
	// Compatibility method name in KVStore; peer mode publishes computed HBM
	// blocks here, never performs unconditional CPU mirroring.
	for _, req := range batch {
		for i, id := range s.RequestMap[req.ID] {
			b := s.Blocks[id]
			end := int64(i+1) * s.BlockSizeTokens
			if end <= min(req.ProgressIndex, req.InputLen()) && b.Hash != "" && int64(len(b.Tokens)) == s.BlockSizeTokens {
				s.ready[id] = b.Hash
				s.lookup[b.Hash] = id
			}
		}
	}
}
func (s *PeerCache) ReleaseKVBlocks(req *sim.Request) {
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
