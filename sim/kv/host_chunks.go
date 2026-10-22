package kv

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// HostPlacementView contains complete immutable objects, never individual GPU
// blocks or future request information. Capacities and usage are object counts.
type HostPlacementView struct {
	NowUS      int64
	ChunkBytes int64
	Incoming   []string
	Pools      map[string]HostPoolSpace
	Candidates []HostObjectCandidate
}
type HostPoolSpace struct{ Capacity, Used int64 }
type HostObjectCandidate struct {
	Key, Pool    string
	LastAccessUS int64
}
type HostPlacementPlan struct {
	// One destination per incoming object. Empty declines the whole batch.
	Destinations []string
	Victims      []string
}
type HostPlacementPolicy interface {
	Place(HostPlacementView) HostPlacementPlan
}

// OrderedHostPlacement fills free space in pool order and reclaims LRU objects
// only in EvictPools. dram-first with CXL-only eviction matches the native
// baseline's retention rule; it does not claim to keep currently hottest DRAM.
type OrderedHostPlacement struct {
	PoolOrder  []string
	EvictPools []string
}

func (p OrderedHostPlacement) Place(v HostPlacementView) HostPlacementPlan {
	free := map[string]int64{}
	for pool, state := range v.Pools {
		free[pool] = state.Capacity - state.Used
	}
	candidates := append([]HostObjectCandidate(nil), v.Candidates...)
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].LastAccessUS != candidates[j].LastAccessUS {
			return candidates[i].LastAccessUS < candidates[j].LastAccessUS
		}
		return candidates[i].Key < candidates[j].Key
	})
	plan := HostPlacementPlan{}
	for range v.Incoming {
		pool := ""
		for _, target := range p.PoolOrder {
			if free[target] > 0 {
				pool = target
				break
			}
		}
		if pool == "" {
			for _, target := range p.EvictPools {
				for i, candidate := range candidates {
					if candidate.Pool == target {
						pool = target
						plan.Victims = append(plan.Victims, candidate.Key)
						candidates = append(candidates[:i], candidates[i+1:]...)
						free[pool]++
						break
					}
				}
				if pool != "" {
					break
				}
			}
		}
		if pool == "" {
			return HostPlacementPlan{} // no partial eviction for a failed batch
		}
		plan.Destinations = append(plan.Destinations, pool)
		free[pool]--
	}
	return plan
}

type hostObject struct {
	key, pool     string
	slot          int64
	entries       []*peerEntry
	lookupReaders int64
	transferRefs  int64
}

func (o *hostObject) storageKey(index int) string { return fmt.Sprintf("%s/%d", o.key, index) }

func (o *hostObject) evictable() bool {
	if o.lookupReaders != 0 || o.transferRefs != 0 {
		return false
	}
	for _, e := range o.entries {
		if !e.ready || e.readers != 0 {
			return false
		}
	}
	return true
}
func (o *hostObject) lastAccess() int64 {
	var at int64
	for _, e := range o.entries {
		at = max(at, e.last)
	}
	return at
}

type hostChunkState struct {
	blocks       int64
	policy       HostPlacementPolicy
	objects      map[string]*hostObject
	pending      map[string]int
	requestHolds map[string][]int64
	inputs       map[string]hostInputIndex
	lookups      map[string]*hostLookup
	ended        map[string]bool
	segmentBytes int64
}

// ConfigureNativeHostGeometry binds the native registration granularity. The
// current dense equal-object adapter requires no padding; variable-size objects
// or unaligned object layouts need a byte arena rather than hidden overcommit.
func (s *PeerCache) ConfigureNativeHostGeometry(alignment, segmentBytes int64) error {
	h := s.fabric.hostChunks
	if h == nil || len(h.objects) != 0 || s.fabric.serial != 0 || alignment <= 0 || alignment&(alignment-1) != 0 || segmentBytes <= 0 ||
		h.blocks*s.fabric.bytes%alignment != 0 {
		return fmt.Errorf("native host geometry requires unused aligned equal-sized chunks")
	}
	for _, p := range s.fabric.pools {
		if p.config.CapacityBlocks > math.MaxInt64/s.fabric.bytes || p.config.CapacityBlocks*s.fabric.bytes%alignment != 0 {
			return fmt.Errorf("host pool bytes overflow or violate alignment")
		}
	}
	h.segmentBytes = segmentBytes
	return nil
}

type hostInputMember struct {
	object string
	index  int
}
type hostInputIndex struct {
	tokens  int
	members map[string]hostInputMember
	objects []string
}

// ConfigureHostChunks changes retention granularity while retaining the BLIS
// allocator and event engine. It does not certify native execution timing.
// The current path is continuous write-through with a fixed HBM LRU-drop rule.
// Host placement policy can be replaced without owning leases or publication.
func (s *PeerCache) ConfigureHostChunks(blocks int64, policy HostPlacementPolicy) error {
	f := s.fabric
	builtin, ok := s.policy.(BuiltinPeerPolicy)
	if blocks <= 0 || blocks > math.MaxInt64/f.bytes || blocks > math.MaxInt64/s.BlockSizeTokens || policy == nil || f.hostChunks != nil || f.native != nil || f.external != nil ||
		len(f.stores) != 1 || len(s.RequestMap) != 0 || len(s.ready) != 0 || f.serial != 0 || f.Pending() != 0 || s.hotprefix != nil ||
		!ok || builtin.Name != "lru_drop" || s.storeSourceReuseEnabled() {
		return fmt.Errorf("host chunks require one idle abstract LRU-drop instance and a placement policy")
	}
	m := f.mechanisms
	if !m.GroupTransfers || m.BackgroundStorePool == "" || m.BackgroundStoreMode != "" && m.BackgroundStoreMode != "continuous" ||
		m.ReserveAtDispatch || m.Online != nil || m.CancelQueuedPromotions {
		return fmt.Errorf("host chunks require grouped continuous stores and immediate reservations")
	}
	if len(s.access) != len(f.pools) {
		return fmt.Errorf("host chunk placement requires access to every configured pool")
	}
	for _, p := range f.pools {
		if len(p.entries) != 0 || p.trackAccess {
			return fmt.Errorf("host chunks own unused pool placement; block eviction overrides are incompatible")
		}
	}
	f.hostChunks = &hostChunkState{blocks: blocks, policy: policy, objects: map[string]*hostObject{}, pending: map[string]int{}, requestHolds: map[string][]int64{}, inputs: map[string]hostInputIndex{}, lookups: map[string]*hostLookup{}, ended: map[string]bool{}}
	if s.prefixCopies == nil {
		s.prefixCopies = map[string][]int64{}
	}
	return nil
}

func (s *PeerCache) restoreIdentity(request, key string) string {
	if s.fabric.hostChunks != nil {
		return request + "\x00" + key
	}
	return key
}

func (f *PeerFabric) joinHostLoads(jobs []*peerJob) {
	if f.hostChunks == nil {
		return
	}
	groups := map[string][]*peerJob{}
	for _, job := range jobs {
		if job.record.Destination == "hbm" {
			groups[job.record.Request] = append(groups[job.record.Request], job)
		}
	}
	for _, members := range groups {
		releaseObjects := f.stores[0].handoffHostLookup(members)
		left := len(members)
		callbacks := make([]func(int64), len(members))
		for i, job := range members {
			callbacks[i] = job.done
			job.done = func(at int64) {
				left--
				if left == 0 {
					for _, done := range callbacks {
						done(at)
					}
					releaseObjects(at)
				}
			}
		}
	}
}

// reserveHostChunks validates the entire policy plan before changing any pool.
// A shared host object has exactly one pool, including its unpublished lifetime.
func (s *PeerCache) reserveHostChunks(chunks [][]*KVBlock, protected map[string]bool) []*hostObject {
	f := s.fabric
	h := f.hostChunks
	view := HostPlacementView{NowUS: s.clock, ChunkBytes: h.blocks * f.bytes, Pools: map[string]HostPoolSpace{}}
	for pool, p := range f.pools {
		view.Pools[pool] = HostPoolSpace{Capacity: p.config.CapacityBlocks / h.blocks, Used: int64(len(p.entries)) / h.blocks}
	}
	seenIncoming := map[string]bool{}
	for _, chunk := range chunks {
		if int64(len(chunk)) != h.blocks {
			panic("incomplete host chunk reservation")
		}
		key := chunk[len(chunk)-1].Hash
		if key == "" || h.objects[key] != nil || seenIncoming[key] {
			panic("host reservation must contain only fresh full chunks")
		}
		seenIncoming[key] = true
		view.Incoming = append(view.Incoming, key)
	}
	allowed := map[string]*hostObject{}
	for key, object := range h.objects {
		if !protected[key] && object.evictable() {
			allowed[key] = object
			view.Candidates = append(view.Candidates, HostObjectCandidate{key, object.pool, object.lastAccess()})
		}
	}
	sort.Slice(view.Candidates, func(i, j int) bool { return view.Candidates[i].Key < view.Candidates[j].Key })
	// Preserve runtime copies even if a policy mutates its detached view.
	incoming := append([]string(nil), view.Incoming...)
	plan := h.policy.Place(view)
	if len(plan.Destinations) == 0 {
		if len(plan.Victims) != 0 {
			panic("declined host placement cannot evict objects")
		}
		return nil
	}
	if len(plan.Destinations) != len(chunks) {
		panic("host placement must cover the complete incoming batch")
	}
	used := map[string]int64{}
	for pool, p := range f.pools {
		used[pool] = int64(len(p.entries)) / h.blocks
	}
	seen := map[string]bool{}
	for _, key := range plan.Victims {
		object := allowed[key]
		if object == nil || seen[key] {
			panic("host placement selected an unavailable or duplicate victim")
		}
		seen[key] = true
		used[object.pool]--
	}
	for _, pool := range plan.Destinations {
		if f.pools[pool] == nil {
			panic("host placement selected an unknown pool")
		}
		used[pool]++
	}
	for pool, count := range used {
		if count > f.pools[pool].config.CapacityBlocks/h.blocks {
			panic("host placement exceeds capacity")
		}
	}
	for _, key := range plan.Victims {
		object := allowed[key]
		for i, e := range object.entries {
			delete(f.pools[object.pool].entries, object.storageKey(i))
			f.emit(PeerRecord{Time: s.clock, Name: "l2_evict", Source: object.pool, Hash: e.hash, Bytes: f.bytes, HostObjects: []string{object.key}, HostChunkBlocks: h.blocks})
		}
		delete(h.objects, key)
	}
	var reserved []*hostObject
	for i, chunk := range chunks {
		object := &hostObject{key: incoming[i], pool: plan.Destinations[i]}
		// Native AddressManager is first-fit in address order. For one fixed,
		// aligned object size this is exactly the lowest unoccupied slot.
		occupied := map[int64]bool{}
		for _, live := range h.objects {
			if live.pool == object.pool {
				occupied[live.slot] = true
			}
		}
		for occupied[object.slot] {
			object.slot++
		}
		for j, b := range chunk {
			e := &peerEntry{hash: b.Hash, tokens: append([]sim.TokenID(nil), b.Tokens...), last: s.clock, object: object}
			object.entries = append(object.entries, e)
			f.pools[object.pool].entries[object.storageKey(j)] = e
			f.emit(PeerRecord{Time: s.clock, Name: "l2_reserve", Destination: object.pool, Hash: b.Hash, Bytes: f.bytes, HostObjects: []string{object.key}, HostChunkBlocks: h.blocks})
		}
		h.objects[object.key] = object
		reserved = append(reserved, object)
	}
	return reserved
}

func (s *PeerCache) storeHostChunks(req *sim.Request) bool {
	return s.storeHostChunksFrom(req, s.RequestMap[req.ID], 0, min(req.ProgressIndex, req.PrefillEnd()), true)
}

// Captured native work may finish after the request's allocation has changed.
// Explicit IDs and a schedule-known range prevent reading new pages. A native
// phased STORE can reserve host space before compute finishes; its physical copy
// must then wait on the producer fence. Legacy calls require already-ready data.
func (s *PeerCache) storeHostChunksFrom(req *sim.Request, ids []int64, startToken, computedEnd int64, requireReady bool) bool {
	f := s.fabric
	h := f.hostChunks
	if f.groupDepth == 0 {
		panic("host stores require an enclosing transfer group")
	}
	var chunks [][]*KVBlock
	protected := map[string]bool{}
	for start := startToken / s.BlockSizeTokens; start+h.blocks <= int64(len(ids)); start += h.blocks {
		if (start+h.blocks)*s.BlockSizeTokens > computedEnd {
			break
		}
		var chunk []*KVBlock
		for _, id := range ids[start : start+h.blocks] {
			b := s.Blocks[id]
			if b.Hash == "" || int64(len(b.Tokens)) != s.BlockSizeTokens || requireReady && s.ready[id] != b.Hash {
				panic("host store source has not been computed and published")
			}
			chunk = append(chunk, b)
		}
		key := chunk[len(chunk)-1].Hash
		protected[key] = true
		if object := h.objects[key]; object != nil {
			if !requireReady {
				for _, entry := range object.entries {
					entry.last = s.clock
				}
			}
			continue
		}
		if requireReady && s.backgroundOffered[req.ID][key] {
			continue
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) == 0 {
		return false
	}
	objects := s.reserveHostChunks(chunks, protected)
	if len(objects) == 0 {
		return false
	}
	start := len(f.groupBuffer)
	for i, object := range objects {
		var path []string
		for _, access := range s.access {
			if access.Pool == object.pool {
				path = access.WritePath
			}
		}
		for j, b := range chunks[i] {
			s.storeReservedCopy(b, object.pool, req.ID, true, path, object.entries[j], true)
		}
		if s.backgroundOffered[req.ID] == nil {
			s.backgroundOffered[req.ID] = map[string]bool{}
		}
		s.backgroundOffered[req.ID][object.key] = true
	}
	// One native MP STORE can contain several media. Nothing is published or
	// released until every member's path completes; GPU/model joining is owned
	// by the transfer executor, not by the placement policy.
	jobs := f.groupBuffer[start:]
	left := len(jobs)
	h.pending[req.ID]++
	callbacks := make([]func(int64), len(jobs))
	for i, job := range jobs {
		callbacks[i] = job.done
		job.done = func(at int64) {
			left--
			if left == 0 {
				for _, done := range callbacks {
					done(at)
				}
				h.pending[req.ID]--
				if h.pending[req.ID] == 0 {
					delete(h.pending, req.ID)
					for _, id := range h.requestHolds[req.ID] {
						s.unpin(s.Blocks[id])
					}
					delete(h.requestHolds, req.ID)
					s.emit("host_store_request_release", req.ID, "", "hbm", "", "all_store_completions")
				}
			}
		}
	}
	return true
}

// Native connector completion keeps the request's full allocation alive after
// its output has finished, not only the blocks read by STORE. Transfers still
// own their own source pins; this additional hold ends after all STORE replies.
func (s *PeerCache) holdHostStoreRequest(req *sim.Request) {
	h := s.fabric.hostChunks
	if h == nil || h.pending[req.ID] == 0 {
		return
	}
	seen := map[int64]bool{}
	for _, id := range h.requestHolds[req.ID] {
		seen[id] = true
	}
	for _, id := range s.RequestMap[req.ID] {
		if !seen[id] {
			s.pin(s.Blocks[id])
			h.requestHolds[req.ID] = append(h.requestHolds[req.ID], id)
		}
	}
	s.emit("host_store_request_hold", req.ID, "", "hbm", "", "store_reply_pending")
}

func (f *PeerFabric) accountHostTransfer(group *peerJob, members []*peerJob) {
	if f.hostChunks == nil {
		return
	}
	objects := map[*hostObject]bool{}
	var ordered []*hostObject
	for _, member := range members {
		if member.hostObject == nil {
			panic("host transfer lacks a reserved immutable object")
		}
		if !objects[member.hostObject] {
			ordered = append(ordered, member.hostObject)
			objects[member.hostObject] = true
		}
	}
	group.units = int64(len(objects)) * f.hostChunks.blocks
	group.record.EffectiveBytes = int64(len(members)) * f.bytes
	group.record.HostChunkBlocks = f.hostChunks.blocks
	for object := range objects {
		group.record.HostObjects = append(group.record.HostObjects, object.key)
	}
	sort.Strings(group.record.HostObjects)
	if f.hostChunks.segmentBytes > 0 {
		group.record.NativeObjects = nil
		for _, object := range ordered {
			bytes := f.hostChunks.blocks * f.bytes
			view := sim.NativeTransferObject{Key: object.key, Pool: object.pool, HostOffset: object.slot * bytes, StagingBytes: bytes}
			first := len(object.entries)
			for _, member := range members {
				if member.hostObject == object {
					view.KernelBytes += f.bytes
					view.HBMBlocks = append(view.HBMBlocks, member.record.HBMBlocks...)
					for i, entry := range object.entries {
						if entry.hash == member.record.Hash {
							first = min(first, i)
						}
					}
				}
			}
			if group.record.Destination == "hbm" {
				view.SkipTokens = int64(first) * f.stores[0].BlockSizeTokens
			}
			for at, left := view.HostOffset, bytes; left > 0; {
				size := min(left, f.hostChunks.segmentBytes-at%f.hostChunks.segmentBytes)
				view.Segments = append(view.Segments, sim.NativeCopySegment{HostOffset: at, Bytes: size})
				at += size
				left -= size
			}
			group.record.NativeObjects = append(group.record.NativeObjects, view)
		}
	}
}

// hostEntryForInput matches the complete host chunk. Matching a GPU-block
// prefix alone is insufficient: two different host objects can contain the
// same early GPU blocks but diverge later in the chunk. Pool entry identities
// therefore include the host object, while HBM content hashes remain unchanged.
func (s *PeerCache) hostInput(request string, input []sim.TokenID) hostInputIndex {
	h := s.fabric.hostChunks
	index, exists := h.inputs[request]
	if !exists || index.tokens != len(input) {
		// Request inputs are immutable; recomputation may extend the visible
		// prefix with already emitted output. Rebuild only on that extension,
		// not once per GPU block or once per scheduler retry.
		index = hostInputIndex{tokens: len(input), members: map[string]hostInputMember{}}
		previous := ""
		var chunk []string
		for i := int64(0); (i+1)*s.BlockSizeTokens <= int64(len(input)); i++ {
			previous = hash.HashBlock(previous, input[i*s.BlockSizeTokens:(i+1)*s.BlockSizeTokens])
			chunk = append(chunk, previous)
			if int64(len(chunk)) == h.blocks {
				index.objects = append(index.objects, previous)
				for offset, member := range chunk {
					index.members[member] = hostInputMember{previous, offset}
				}
				chunk = chunk[:0]
			}
		}
		h.inputs[request] = index
	}
	return index
}

func (s *PeerCache) hostEntryForInput(request, key string, input []sim.TokenID) (*peerEntry, *PeerAccess) {
	h := s.fabric.hostChunks
	index := s.hostInput(request, input)
	member, exists := index.members[key]
	if !exists || h.objects[member.object] == nil {
		return nil, nil
	}
	object := h.objects[member.object]
	for j := range s.access {
		if s.access[j].Pool == object.pool {
			return object.entries[member.index], &s.access[j]
		}
	}
	panic("resident host object is inaccessible")
}

func (s *PeerCache) findForInput(request, key string, input []sim.TokenID) (*peerEntry, *PeerAccess) {
	if s.fabric.hostChunks == nil {
		return s.find(key)
	}
	e, access := s.hostEntryForInput(request, key, input)
	if e != nil && e.ready {
		return e, access
	}
	return nil, nil
}
