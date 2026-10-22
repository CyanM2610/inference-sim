package kv

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

// LMCacheMPConnector inherits KVConnectorBase_V1.requires_kv_delivery.
// This native hybrid path uses kv_both, whose is_kv_producer is true.
func (s *PeerCache) NativeRequiresKVDelivery() bool { return true }

// Promotion is an explicit connector-control extension. Ordinary request
// scheduling stays native; idle control iterations pay the same empty worker
// plan and positive host schedule/consume services. The native connector keeps
// polling throughout transfer and receipt adoption, including after a worker
// has claimed the receipt but before FIFO consumption publishes the target.
func (s *PeerCache) NativeControlPending() bool {
	if !s.nativePhaseTransfers {
		return false
	}
	if s.nativeControl != nil {
		return true
	}
	// This ownership already spans reservation through adoption. Prepared and
	// receipt queues alone miss both the copy and claimed-receipt intervals.
	prefix := s.restoreIdentity("@promotion", "")
	for identity := range s.restoring {
		if strings.HasPrefix(identity, prefix) {
			return true
		}
	}
	return false
}

// Native BlockPool.free_blocks preserves reusable cached pages at the tail,
// but prepends unhashed pages for immediate reuse. ids are in allocation order;
// native request frees traverse them in reverse, preserving each partition's
// order within this one release call. Legacy paths retain their prior FIFO.
func (s *PeerCache) releaseNativeReferences(ids []int64) {
	var uncached []*KVBlock
	for i := len(ids) - 1; i >= 0; i-- {
		b := s.Blocks[ids[i]]
		if b.RefCount <= 0 {
			panic("native release of an unowned page")
		}
		b.RefCount--
		if b.RefCount != 0 {
			continue
		}
		b.InUse = false
		if b.Hash == "" {
			uncached = append(uncached, b)
		} else {
			s.appendToFreeList(b)
		}
	}
	// Prepend the whole ordered partition, not each member in forward order.
	for i := len(uncached) - 1; i >= 0; i-- {
		b := uncached[i]
		b.PrevFree, b.NextFree = nil, s.FreeHead
		if s.FreeHead != nil {
			s.FreeHead.PrevFree = b
		} else {
			s.FreeTail = b
		}
		s.FreeHead = b
		s.FreeBlockCnt++
	}
}

func (s *PeerCache) EnableNativeAsyncKV() error {
	if s.nativeAsync || s.fabric.hostChunks == nil || s.fabric.phases != nil || s.batchPrefixes != nil || len(s.RequestMap) != 0 || s.fabric.Pending() != 0 {
		return fmt.Errorf("native async KV requires unused host chunks without the legacy batch executor")
	}
	s.nativeAsync = true
	s.nativePending = map[string]int{}
	s.nativeLastStep = map[string]int64{}
	s.nativeLoadPending = map[string]bool{}
	s.nativeLoadTargets = map[string][]int64{}
	s.nativeStored = map[string]nativeStorePosition{}
	// Full external chunks may include the last prompt block. Recompute one
	// token for logits while retaining that physical block, as native does.
	s.restoreControl = true
	s.restoreLimits = map[string]int64{}
	s.restoredPrefixBlocks = map[string]int64{}
	s.nativeRestoreAdmission = true
	return nil
}

func (s *PeerCache) EnableNativePhaseTransfers(executor *sim.NativeExecutor, resource string) error {
	if s.nativeAsync || s.nativePhaseTransfers || s.fabric.Pending() != 0 || len(s.RequestMap) != 0 || executor == nil || resource == "" || s.fabric.mechanisms.CompletionUS != 0 {
		return fmt.Errorf("native phase transfers require an unused native async store")
	}
	if err := s.EnableNativeAsyncKV(); err != nil {
		return err
	}
	s.nativePhaseTransfers = true
	s.nativeExecutor, s.nativeCopyResource = executor, resource
	s.nativePageAccess = map[int64]nativePendingPageAccess{}
	return nil
}

func (s *PeerCache) ReserveNativeRange(req *sim.Request, start, end int64, cached []int64) bool {
	if !s.nativeAsync || s.nativeReserving || start < 0 || end <= start || start >= req.PrefillEnd() && end != start+1 {
		panic("invalid native reservation")
	}
	shadow := *req
	shadow.ProgressIndex = start
	// Allocate capacity for decode without exposing offline output identities.
	shadow.OutputTokens = req.OutputTokens[:req.EmittedTokens()]
	s.nativeReserving = true
	defer func() { s.nativeReserving = false }()
	if !s.AllocateKVBlocks(&shadow, start, end, cached) {
		return false
	}
	// Native APC registers finalized token hashes at allocate_slots(), before
	// device completion. Later uses are ordered on the same worker stream; host
	// A physical STORE waits for the captured producer fence. Unknown placeholders are
	// not indexed. This differs from the legacy completion-only APC mode.
	known := min(end, req.InputLen()+req.EmittedTokens())
	if known < req.InputLen() {
		known = min(end, req.InputLen())
	}
	s.nativeHashAndPublish(req, s.RequestMap[req.ID], known, true)
	return true
}

type nativeCapturedWork struct {
	work                 sim.NativeWork
	ids                  []int64
	storeStart, storeEnd int64
}

type nativeStorePosition struct {
	generation uint64
	tokens     int64
}

type nativeRetiredAllocation struct {
	request string
	ids     []int64
	fence   int64
}

// Pinned vLLM fences an in-flight request free by the most recent GLOBAL
// scheduled computing step, which may be newer than that request's last step.
// Every detached allocation owns a distinct reference, including shared APC
// pages and later allocations with the same request ID.
func (s *PeerCache) retireNativeAllocation(req *sim.Request) {
	ids := s.RequestMap[req.ID]
	if len(ids) == 0 || s.nativeLastStep[req.ID] <= s.nativeProcessedStep {
		return
	}
	retired := nativeRetiredAllocation{request: req.ID, ids: append([]int64(nil), ids...), fence: s.nativeScheduledStep}
	for _, id := range retired.ids {
		s.pin(s.Blocks[id])
	}
	s.nativeDeferred = append(s.nativeDeferred, retired)
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "native_allocation_retired", Instance: s.id, Request: req.ID,
		HBMBlocks: append([]int64(nil), retired.ids...), Counters: map[string]int64{
			"fence_step": retired.fence, "last_request_step": s.nativeLastStep[req.ID], "processed_step": s.nativeProcessedStep}})
}

func (s *PeerCache) drainNativeRetirements() {
	for len(s.nativeDeferred) > 0 && s.nativeDeferred[0].fence <= s.nativeProcessedStep {
		retired := s.nativeDeferred[0]
		s.releaseNativeReferences(retired.ids)
		s.nativeDeferred[0] = nativeRetiredAllocation{}
		s.nativeDeferred = s.nativeDeferred[1:]
		s.fabric.emit(PeerRecord{Time: s.clock, Name: "native_allocation_retirement_released", Instance: s.id, Request: retired.request,
			HBMBlocks: append([]int64(nil), retired.ids...), Counters: map[string]int64{
				"fence_step": retired.fence, "processed_step": s.nativeProcessedStep}})
	}
	if len(s.nativeDeferred) == 0 {
		s.nativeDeferred = nil
	}
}

type nativeCapturedBatch struct {
	store                *PeerCache
	works                []nativeCapturedWork
	deviceDone, consumed bool
	consumeStarted       bool
	step                 int64
	posted, polled       bool
	loads                []*peerJob
	receives             []string
	promotions           []int64
	control              *nativePromotion
	controlReceipt       *nativePromotion
}

// The physical STORE read / LOAD write outlives allocator ownership. Keep the
// exact submitted pages even if their logical hash or request later changes.
type nativePendingPageAccess struct {
	ids   []int64
	write bool
	fence *sim.NativeFence
}

func (b *nativeCapturedBatch) KVAccessPlan() sim.NativeKVAccessPlan {
	s := b.store
	reads, writes := map[int64]bool{}, map[int64]bool{}
	for _, captured := range b.works {
		for i, id := range captured.ids {
			reads[id] = true
			if int64(i) >= captured.work.Start/s.BlockSizeTokens {
				writes[id] = true
			}
		}
	}
	// A connector-only iteration can overwrite a page by LOAD without forward.
	for _, j := range b.loads {
		for _, id := range j.record.HBMBlocks {
			writes[id] = true
		}
	}
	result := sim.NativeKVAccessPlan{}
	for id := range reads {
		result.ReadBlocks = append(result.ReadBlocks, id)
	}
	for id := range writes {
		result.WriteBlocks = append(result.WriteBlocks, id)
	}
	sort.Slice(result.ReadBlocks, func(i, j int) bool { return result.ReadBlocks[i] < result.ReadBlocks[j] })
	sort.Slice(result.WriteBlocks, func(i, j int) bool { return result.WriteBlocks[i] < result.WriteBlocks[j] })
	for transaction, pending := range s.nativePageAccess {
		if pending.fence.Ready() {
			continue
		}
		var ids []int64
		for _, id := range pending.ids {
			if writes[id] || pending.write && reads[id] {
				ids = append(ids, id)
			}
		}
		if len(ids) > 0 {
			result.Dependencies = append(result.Dependencies, sim.NativeKVAccessDependency{Transaction: transaction, HBMBlocks: ids, PendingWrite: pending.write, Fence: pending.fence})
		}
	}
	sort.Slice(result.Dependencies, func(i, j int) bool { return result.Dependencies[i].Transaction < result.Dependencies[j].Transaction })
	return result
}

func (s *PeerCache) CaptureNativeBatch(work []sim.NativeWork) sim.NativeBatchKV {
	if !s.nativeAsync {
		panic("native async KV is disabled")
	}
	b := &nativeCapturedBatch{store: s, loads: append(s.nativePrepared, s.nativePromotionPrepared...)}
	s.nativePrepared = nil
	s.nativePromotionPrepared = nil
	if len(work) > 0 {
		s.nativeScheduledStep++
		b.step = s.nativeScheduledStep
	}
	for _, w := range work {
		s.nativeLastStep[w.Request.ID] = b.step
		n := (w.End + s.BlockSizeTokens - 1) / s.BlockSizeTokens
		ids := s.RequestMap[w.Request.ID]
		if n <= 0 || n > int64(len(ids)) {
			panic("native work has no reserved target")
		}
		captured := append([]int64(nil), ids[:n]...)
		for _, id := range captured {
			s.pin(s.Blocks[id])
			s.nativeBatchPins++
		}
		position, exists := s.nativeStored[w.Request.ID]
		if !exists || position.generation != w.Generation {
			position = nativeStorePosition{generation: w.Generation}
			if w.Generation == 0 {
				if lookup := s.fabric.hostChunks.lookups[w.Request.ID]; lookup != nil {
					position.tokens = int64(len(lookup.prefix)) * s.fabric.hostChunks.blocks * s.BlockSizeTokens
				}
			}
		}
		chunkTokens := s.fabric.hostChunks.blocks * s.BlockSizeTokens
		storeEnd := min(w.End, w.KnownTokens) / chunkTokens * chunkTokens
		b.works = append(b.works, nativeCapturedWork{work: w, ids: captured, storeStart: position.tokens, storeEnd: storeEnd})
		// Pinned connector advances num_stored_blocks when metadata is built,
		// even if server admission later rejects STORE. No implicit retry is
		// created merely because a later batch has the same completed prefix.
		position.tokens = max(position.tokens, storeEnd)
		s.nativeStored[w.Request.ID] = position
		s.nativePending[w.Request.ID]++
	}
	b.control = s.prepareNativePromotion()
	return b
}

// Resolve only finalized input/observed output identities at schedule time,
// or device-produced identities at completion. No caller publishes beyond end.
func (s *PeerCache) nativeHashAndPublish(req *sim.Request, ids []int64, end int64, publish bool) {
	previous := ""
	for i, id := range ids {
		if int64(i+1)*s.BlockSizeTokens > end {
			break
		}
		b := s.Blocks[id]
		for j := range b.Tokens {
			index := int64(i)*s.BlockSizeTokens + int64(j)
			if index < req.InputLen() {
				b.Tokens[j] = req.InputTokens[index]
			} else {
				index -= req.InputLen()
				if index >= int64(len(req.OutputTokens)) {
					panic("native hash exceeded available token identity")
				}
				b.Tokens[j] = req.OutputTokens[index]
			}
		}
		hash := s.hashBlock(previous, b.Tokens)
		if b.Hash != hash {
			if _, ready := s.ready[id]; ready {
				panic("native finalized APC content changed")
			}
			if s.HashToBlock[b.Hash] == id {
				delete(s.HashToBlock, b.Hash)
			}
			b.Hash = hash
		}
		previous = hash
		if publish {
			s.publishReadyCopy(b)
		}
	}
}

func (b *nativeCapturedBatch) DeviceComplete(now int64) {
	b.Computed(now)
	b.PostForward(now, nil)
	b.Poll(now)
}

func (b *nativeCapturedBatch) Computed(now int64) {
	if b.deviceDone || b.consumed {
		panic("duplicate native device completion")
	}
	b.deviceDone = true
	s := b.store
	s.SetClock(now)
	for _, captured := range b.works {
		w := captured.work
		// Device execution may run ahead of host EOS observation. Only real token
		// identities can enter the reusable directory; extra work still pays cost.
		end := min(w.End, w.Request.InputLen()+int64(len(w.Request.OutputTokens)))
		s.nativeHashAndPublish(w.Request, captured.ids, end, false)
	}
}

func (b *nativeCapturedBatch) PostForward(now int64, producer *sim.NativeFence) {
	s := b.store
	if b.posted || b.consumed || s.nativePhaseTransfers && producer == nil {
		panic("invalid native post-forward")
	}
	b.posted = true
	s.SetClock(now)
	if len(s.nativePrepared) > 0 {
		panic("uncaptured native LOAD metadata")
	}
	// Sources are schedule-known immutable ranges. Physical copies wait on the
	// producer's device fence even when host post_forward precedes GPU completion.
	for _, captured := range b.works {
		w := captured.work
		s.fabric.beginTransferGroup()
		s.storeHostChunksFrom(w.Request, captured.ids, captured.storeStart, captured.storeEnd, false)
		s.fabric.endTransferGroup(now)
		if w.Request.State == sim.StateTimedOut || w.Request.State == sim.StateCompleted {
			// A cancelled request can acquire STORE obligations after its host END.
			// Keep all captured pages until those STORE replies drain.
			s.holdCapturedHostStore(w.Request.ID, captured.ids)
		}
	}
	if b.control != nil {
		s.submitNativePromotion(b.control)
	}
	jobs := append(b.loads, s.nativePrepared...)
	b.loads = nil
	s.nativePrepared = nil
	for _, j := range jobs {
		s.fabric.nativePrepared--
		j.record.Time, j.record.Name = now, "transfer_queued"
		s.fabric.emit(j.record)
		f := s.fabric
		f.nativeInFlight++
		if s.nativeExecutor.HasTransferModel() {
			if len(j.record.NativeObjects) == 0 {
				panic("native staged transfer requires explicit host geometry")
			}
			direction := "store"
			if j.record.Destination == "hbm" {
				direction = "load"
			}
			objects := append([]sim.NativeTransferObject(nil), j.record.NativeObjects...)
			for i := range objects {
				objects[i].HBMBlocks = append([]int64(nil), objects[i].HBMBlocks...)
				objects[i].Segments = append([]sim.NativeCopySegment(nil), objects[i].Segments...)
			}
			started := int64(-1)
			fence := s.nativeExecutor.SubmitTransfer(now, j.record.Transaction, sim.NativeTransferShape{Direction: direction, Objects: objects}, producer,
				func(at int64) { started = at; j.record.Time, j.record.Name = at, "transfer_start"; f.emit(j.record) },
				func(at int64) {
					record := j.record
					record.Time, record.Name, record.Duration = at, "transfer_cuda_complete", at-started
					f.emit(record)
				},
				func(at int64) {
					delete(s.nativePageAccess, j.record.Transaction)
					f.nativeInFlight--
					j.record.Time, j.record.Name, j.record.Duration = at, "transfer_end", at-started
					f.emit(j.record)
					j.done(at)
					f.notifyDecisionTransfer(at, j)
					f.wakeStores(at)
				})
			s.nativePageAccess[j.record.Transaction] = nativePendingPageAccess{ids: append([]int64(nil), j.record.HBMBlocks...), write: direction == "load", fence: fence}
			continue
		}
		duration := f.durationBytes(j.path, max(1, j.units)*f.bytes)
		s.nativeExecutor.SubmitService(now, fmt.Sprintf("kv_copy_%d", j.record.Transaction), s.nativeCopyResource, "native-kv-stream", duration, producer,
			func(at int64) {
				for _, id := range j.path {
					f.busy[id] = true
					f.busyUntil[id] = at + duration
				}
				j.record.Time, j.record.Name, j.record.Duration = at, "transfer_start", duration
				f.emit(j.record)
			}, func(at int64) {
				for _, id := range j.path {
					f.busy[id] = false
				}
				f.nativeInFlight--
				j.record.Time, j.record.Name = at, "transfer_end"
				f.emit(j.record)
				j.done(at)
				f.notifyDecisionTransfer(at, j)
				f.wakeStores(at)
			})
	}
}

func (b *nativeCapturedBatch) Poll(now int64) {
	if !b.posted || b.polled || b.consumed {
		panic("invalid native receipt poll")
	}
	b.polled = true
	s := b.store
	s.SetClock(now)
	b.promotions = s.nativePromotionReceipts
	s.nativePromotionReceipts = nil
	if c := s.nativeControl; c != nil && c.submitted && !c.claimed && c.completed == len(c.ids)-c.cached {
		c.claimed = true
		b.controlReceipt = c
		s.emit("native_control_receipt", c.id, "", "host", "hbm", "worker_poll")
	}
	for request, pending := range s.nativeLoadPending {
		if pending && s.readPending[request] == 0 && len(s.nativeLoadTargets[request]) > 0 {
			b.receives = append(b.receives, request)
		}
	}
	sort.Strings(b.receives)
}

func (s *PeerCache) holdCapturedHostStore(request string, ids []int64) {
	h := s.fabric.hostChunks
	if h.pending[request] == 0 {
		return
	}
	seen := map[int64]bool{}
	for _, id := range h.requestHolds[request] {
		seen[id] = true
	}
	for _, id := range ids {
		if !seen[id] {
			s.pin(s.Blocks[id])
			h.requestHolds[request] = append(h.requestHolds[request], id)
			seen[id] = true
		}
	}
}

func (b *nativeCapturedBatch) BeforeConsume(now int64) {
	if !b.deviceDone || !b.polled || b.consumeStarted || b.consumed {
		panic("invalid native consumption boundary")
	}
	b.consumeStarted = true
	s := b.store
	s.SetClock(now)
	if b.step > 0 {
		if b.step != s.nativeProcessedStep+1 {
			panic("native computing outputs consumed out of order")
		}
		s.nativeProcessedStep = b.step
	}
	s.fabric.emit(PeerRecord{Time: now, Name: "native_output_consumption_begin", Instance: s.id,
		Counters: map[string]int64{"computing_step": b.step, "processed_step": s.nativeProcessedStep, "scheduled_step": s.nativeScheduledStep}})
	// Captures are a simulator safety reference, not an extra native block-pool
	// owner. Remove them BEFORE native retirement references so that the physical
	// free queue follows retirement order rather than captured-work iteration.
	// Live allocations and pending transfer holds still own their own references.
	for _, captured := range b.works {
		s.releaseNativeReferences(captured.ids)
		s.nativeBatchPins -= int64(len(captured.ids))
	}
	if b.step > 0 {
		s.drainNativeRetirements()
	}
}

func (b *nativeCapturedBatch) Consumed(now int64) {
	if !b.consumeStarted || b.consumed {
		panic("native batch consumed without its fence boundary or twice")
	}
	b.consumed = true
	s := b.store
	s.SetClock(now)
	if b.controlReceipt != nil {
		s.adoptNativePromotion(b.controlReceipt)
	}
	for _, id := range b.promotions {
		block := s.Blocks[id]
		s.publishReadyCopy(block)
		delete(s.restoring, s.restoreIdentity("@promotion", block.Hash))
		s.promoted[id] = true
		s.emit("hbm_publish", "@promotion", block.Hash, "host", "hbm", "native_promotion_adopted")
		s.releaseNativeReferences([]int64{id})
	}
	if len(b.promotions) > 0 {
		s.emit("native_promotion_adopted", "@promotion", "", "host", "hbm", "fifo_output_consumption")
		b.promotions = nil
	}
	for _, request := range b.receives {
		if !s.nativeLoadPending[request] {
			continue
		}
		for _, id := range s.nativeLoadTargets[request] {
			block := s.Blocks[id]
			s.publishReadyCopy(block)
			s.emit("hbm_publish", request, block.Hash, "host", "hbm", "native_receive_adopted")
		}
		delete(s.nativeLoadTargets, request)
		delete(s.nativeLoadPending, request)
		s.emit("native_receive_adopted", request, "", "host", "hbm", "fifo_output_consumption")
	}
	for _, captured := range b.works {
		w := captured.work
		// Never expose a token identity which host output consumption has not
		// finalized, including a cancelled batch's unobserved generated suffix.
		end := min(w.End, w.Request.InputLen()+w.Request.EmittedTokens())
		s.nativeHashAndPublish(w.Request, captured.ids, end, true)
		s.nativePending[w.Request.ID]--
		if s.nativePending[w.Request.ID] == 0 {
			delete(s.nativePending, w.Request.ID)
			if w.Request.State == sim.StateCompleted || w.Request.State == sim.StateTimedOut {
				delete(s.nativeStored, w.Request.ID)
				delete(s.nativeLastStep, w.Request.ID)
			}
		}
	}
}
