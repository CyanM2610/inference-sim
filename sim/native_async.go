package sim

import (
	"fmt"
	"math"
	"sort"
)

const NativeAsyncRevision = "98dff2a81d747d1dba01a47f939f48c3526d4206"

type nativeRequestProgress struct {
	generation              uint64
	scheduled, placeholders int64
	pending                 int
}

type nativeQueuedBatch struct {
	id       int64
	finished []string
	work     []NativeWork
	shape    []NativeBatchShape
	kv       NativeBatchKV
	ready    bool
}

type nativeAsyncRuntime struct {
	s                      *Simulator
	store                  NativeAsyncKVStore
	cost                   NativeHostCost
	executor               *NativeExecutor
	depth                  int
	requests               map[string]*nativeRequestProgress
	queue                  []*nativeQueuedBatch
	skipped                []string
	busy, waiting          bool
	workerFree, outputFree int64
	serial                 int64
	finished               []string
	accountedAt            int64
}

// NativeAsyncEvent exposes actual runtime transitions, never trace-forced times.
// Shape is a submission snapshot, so later formation cannot rewrite past work.
type NativeAsyncEvent struct {
	At               int64                `json:"time_us"`
	Name             string               `json:"name"`
	BatchID          int64                `json:"batch_id"`
	Shape            []NativeBatchShape   `json:"shape,omitempty"`
	QueueDepth       int                  `json:"queue_depth"`
	Phase            string               `json:"phase,omitempty"`
	Resource         string               `json:"resource,omitempty"`
	DurationUS       int64                `json:"duration_us,omitempty"`
	Execution        *NativeExecutionPlan `json:"execution,omitempty"`
	KVAccess         *NativeKVAccessPlan  `json:"kv_access,omitempty"`
	Transaction      int64                `json:"transaction,omitempty"`
	Transfer         *NativeTransferPlan  `json:"transfer,omitempty"`
	TransferShape    *NativeTransferShape `json:"transfer_shape,omitempty"`
	ProducerBatch    int64                `json:"producer_batch,omitempty"`
	ProducerPhase    string               `json:"producer_phase,omitempty"`
	Request          string               `json:"request,omitempty"`
	CoverageGap      string               `json:"coverage_gap,omitempty"`
	Generation       uint64               `json:"generation,omitempty"`
	FinishedRequests []string             `json:"finished_requests,omitempty"`
	run              func()
}

func (e *NativeAsyncEvent) Timestamp() int64 { return e.At }
func (e *NativeAsyncEvent) Priority() int    { return PriorityAdapterLoad }
func (e *NativeAsyncEvent) Execute(*Simulator) {
	if e.run != nil {
		e.run()
	}
}

// EnableNativeAsync installs the dense single-instance native FCFS execution
// path. Explicit positive cost services avoid a hidden zero-cost polling loop.
// The caller owns calibration/coverage; enabling this is not a calibration PASS.
func (s *Simulator) EnableNativeAsync(depth int, cost NativeHostCost) error {
	store, ok := s.KVCache.(NativeAsyncKVStore)
	if !ok || cost == nil || depth < 2 || s.nativeAsync != nil || s.stepCount != 0 || !s.batchCompletionEvents || s.specEnabled || s.residentAdapters != nil || s.decision != nil || s.hostServices != nil || s.postStep != nil || s.latencyModel.OutputTokenProcessingTime() != 0 || s.latencyModel.PostDecodeFixedOverhead() != 0 {
		return fmt.Errorf("native async requires an idle dense instance, native KV, depth >=2 and explicit costs")
	}
	phase, phased := cost.(NativePhaseCost)
	if _, coarse := cost.(NativeAsyncCost); !phased && !coarse {
		return fmt.Errorf("native execution model is missing")
	}
	n := &nativeAsyncRuntime{s: s, store: store, cost: cost, depth: depth, requests: map[string]*nativeRequestProgress{}}
	if phased {
		if _, ok := store.(NativePhasedKVStore); !ok {
			return fmt.Errorf("native phase model requires a phase-aware KV store")
		}
		var err error
		n.executor, err = newNativeExecutor(n, phase.ExecutionResources())
		if err != nil {
			return err
		}
		if _, ok := n.executor.busy[phase.CopyResource()]; !ok {
			return fmt.Errorf("native copy resource is not configured")
		}
	}
	if phased {
		if err := store.(NativePhasedKVStore).EnableNativePhaseTransfers(n.executor, phase.CopyResource()); err != nil {
			return err
		}
	} else if err := store.EnableNativeAsyncKV(); err != nil {
		return err
	}
	s.scheduler = &FCFSScheduler{}
	s.nativeAsync = n
	store.BindEvents(s.Schedule, s.ScheduleStepIfIdle)
	return nil
}

func (n *nativeAsyncRuntime) account(now int64) {
	if now < n.accountedAt {
		panic("native clock regressed")
	}
	used := n.store.UsedBlocks()
	n.s.Metrics.KVBlocksUsed += float64(used) * float64(now-n.accountedAt)
	n.s.Metrics.PeakKVBlocksUsed = max(n.s.Metrics.PeakKVBlocksUsed, used)
	n.accountedAt = now
}

func (n *nativeAsyncRuntime) cancelled(r *Request) {
	n.finished = append(n.finished, r.ID)
	if p := n.requests[r.ID]; p != nil && p.pending == 0 {
		delete(n.requests, r.ID)
	}
}

func nativeAfter(now, duration int64, positive bool) int64 {
	if duration < 0 || positive && duration == 0 || duration > math.MaxInt64-now {
		panic("invalid or overflowing native service duration")
	}
	return now + duration
}

func (n *nativeAsyncRuntime) event(at int64, name string, b *nativeQueuedBatch, run func()) *NativeAsyncEvent {
	e := &NativeAsyncEvent{At: at, Name: name, QueueDepth: len(n.queue)}
	e.run = func() {
		if run != nil {
			run()
		}
		e.QueueDepth = len(n.queue)
	}
	if b != nil {
		e.BatchID = b.id
		e.FinishedRequests = append([]string(nil), b.finished...)
		e.Shape = append([]NativeBatchShape(nil), b.shape...)
	}
	n.s.Schedule(e)
	return e
}

func (n *nativeAsyncRuntime) hasRequests() bool {
	return n.store.NativeControlPending() || len(n.finished) > 0 || n.s.WaitQ.Len() > 0 || n.s.RunningBatch != nil && len(n.s.RunningBatch.Requests) > 0
}

func (n *nativeAsyncRuntime) progress(r *Request) *nativeRequestProgress {
	p := n.requests[r.ID]
	if p == nil {
		if r.MaxOutputLen <= 0 || r.Adapter != "" || r.IsDecodeSubRequest || r.IsMultimodal() {
			panic("native async requires a dense request with a public positive output budget")
		}
		p = &nativeRequestProgress{}
		n.requests[r.ID] = p
	}
	return p
}

func (n *nativeAsyncRuntime) next(now int64) {
	n.s.stepEvent = nil
	if n.busy {
		panic("native next while host service is busy")
	}
	if len(n.queue) >= n.depth || !n.hasRequests() && len(n.queue) > 0 {
		n.waiting = true
		n.tryConsume(now)
		return
	}
	if n.hasRequests() {
		e := &StepEvent{time: now}
		n.s.stepEvent = e
		n.s.Schedule(e)
	}
}

func (n *nativeAsyncRuntime) step(now int64) {
	_, fcfs := n.s.scheduler.(*FCFSScheduler)
	if !fcfs || n.s.decision != nil || n.s.hostServices != nil || n.s.postStep != nil {
		panic("native async cannot silently ignore an incompatible scheduler/service configuration")
	}
	if n.busy || n.waiting {
		return
	}
	if len(n.queue) >= n.depth {
		n.waiting = true
		n.tryConsume(now)
		return
	}
	if !n.hasRequests() {
		n.next(now)
		return
	}
	n.s.KVCache.SetClock(now)
	n.s.stepCount++
	n.busy = true
	running := 0
	if n.s.RunningBatch != nil {
		running = len(n.s.RunningBatch.Requests)
	}
	end := nativeAfter(now, n.cost.ScheduleUS(n.s.WaitQ.Len(), running), true)
	b := n.form(now)
	n.s.stepEvent = n.event(end, "native_async_scheduled", b, func() {
		n.queue = append(n.queue, b)
		if n.executor != nil {
			n.executePhases(end, b)
			return
		}
		cost := n.cost.(NativeAsyncCost)
		start := max(end, n.workerFree)
		deviceEnd := nativeAfter(start, cost.WorkerUS(append([]NativeBatchShape(nil), b.shape...)), true)
		n.workerFree = deviceEnd
		outputEnd := nativeAfter(max(deviceEnd, n.outputFree), cost.OutputUS(append([]NativeBatchShape(nil), b.shape...)), false)
		n.outputFree = outputEnd
		n.event(start, "native_async_worker_start", b, nil)
		n.event(deviceEnd, "native_async_device_complete", b, func() { b.kv.DeviceComplete(deviceEnd) })
		n.event(outputEnd, "native_async_output_ready", b, func() {
			b.ready = true
			if n.waiting && !n.busy {
				n.tryConsume(outputEnd)
			}
		})
		n.busy = false
		n.next(end)
	})
}

func (n *nativeAsyncRuntime) executePhases(now int64, b *nativeQueuedBatch) {
	p := n.cost.(NativePhaseCost).ExecutionPlan(append([]NativeBatchShape(nil), b.shape...))
	kv, ok := b.kv.(NativePhasedBatchKV)
	if !ok {
		panic("phase-aware KV capture missing")
	}
	access := kv.KVAccessPlan()
	record := n.event(now, "native_execution_plan", b, nil)
	record.Execution, record.KVAccess = &p, &access
	fences := map[string]*NativeFence{p.ProducerReady: {}}
	callbacks := map[string]func(int64){}
	// Several milestones may intentionally be the same host-only empty step.
	add := func(name string, callback func(int64)) {
		previous := callbacks[name]
		callbacks[name] = func(at int64) {
			if previous != nil {
				previous(at)
			}
			callback(at)
		}
	}
	add(p.Computed, kv.Computed)
	add(p.PostForward, func(at int64) { kv.PostForward(at, fences[p.ProducerReady]) })
	add(p.Poll, kv.Poll)
	add(p.OutputReady, func(at int64) {
		b.ready = true
		if n.waiting && !n.busy {
			n.tryConsume(at)
		}
	})
	add(p.HostReturned, func(at int64) { n.busy = false; n.next(at) })
	n.executor.submit(now, b, p, callbacks, fences, access.Dependencies)
}

func (n *nativeAsyncRuntime) tryConsume(now int64) {
	if n.busy || len(n.queue) == 0 {
		return
	}
	b := n.queue[0]
	if !b.ready {
		// Marker only; the already-scheduled worker/output event owns the wake.
		n.s.stepEvent = &NativeAsyncEvent{At: now, Name: "native_async_await_output", BatchID: b.id}
		return
	}
	n.busy = true
	end := nativeAfter(now, n.cost.ConsumeUS(len(b.work)), true)
	n.s.stepEvent = n.event(end, "native_async_consumed", b, func() {
		n.s.KVCache.SetClock(end)
		b.kv.BeforeConsume(end)
		for _, w := range b.work {
			n.consumeWork(w, end, b)
		}
		b.kv.Consumed(end)
		n.queue = n.queue[1:]
		n.waiting = false
		n.busy = false
		n.next(end)
	})
}

func (n *nativeAsyncRuntime) preempt(r *Request) {
	p := n.progress(r)
	// Preserve observed output; the connector contract decides whether stale
	// output is dropped and whether its drain limits waiting admission.
	r.preserveOutputForRecompute()
	n.s.KVCache.ReleaseKVBlocks(r) // StateRunning preserves consumed host session.
	p.generation++
	p.scheduled = 0
	p.placeholders = 0
	r.State = StateQueued
	r.ProgressIndex = 0
	r.NumNewTokens = 0
	n.s.WaitQ.PrependFront(r)
	n.s.Metrics.PreemptionCount++
	delete(n.s.reqNumComputedTokens, r.ID)
}

func (n *nativeAsyncRuntime) form(now int64) *nativeQueuedBatch {
	s := n.s
	if s.RunningBatch == nil {
		s.RunningBatch = &Batch{}
	}
	n.serial++
	// Native Scheduler.has_finished_requests keeps one metadata iteration alive
	// after final output. Snapshot IDs here; later completions belong to the
	// next iteration even if this batch has not reached the worker yet.
	b := &nativeQueuedBatch{id: n.serial, finished: n.finished}
	n.finished = nil
	budget := s.maxNumBatchedTokens
	preempted := false
	for index := 0; index < len(s.RunningBatch.Requests) && budget > 0; index++ {
		r := s.RunningBatch.Requests[index]
		p := n.progress(r)
		if p.placeholders > 0 && p.scheduled+2-p.placeholders >= r.InputLen()+int64(r.MaxOutputLen) {
			continue
		}
		known := r.InputLen() + r.EmittedTokens()
		grant := nativeTokenGrant(known+p.placeholders-p.scheduled, budget, s.longPrefillTokenThreshold)
		if s.maxModelLen > 0 {
			grant = min(grant, max(0, s.maxModelLen-1-p.scheduled))
		}
		if grant == 0 {
			continue
		}
		ok := false
		for len(s.RunningBatch.Requests) > 0 {
			if n.store.ReserveNativeRange(r, p.scheduled, p.scheduled+grant, nil) {
				ok = true
				break
			}
			last := len(s.RunningBatch.Requests) - 1
			victim := s.RunningBatch.Requests[last]
			s.RunningBatch.Requests = s.RunningBatch.Requests[:last]
			// A victim already selected during this formation loses that work.
			for i, w := range b.work {
				if w.Request == victim {
					budget += w.End - w.Start
					b.work = append(b.work[:i], b.work[i+1:]...)
					break
				}
			}
			n.preempt(victim)
			preempted = true
			if victim == r {
				break
			}
		}
		if !ok {
			break
		}
		b.work = append(b.work, NativeWork{Request: r, Generation: p.generation, Start: p.scheduled, End: p.scheduled + grant, KnownTokens: known})
		budget -= grant
	}
	// Keep skipped_waiting's persistent ordering without exposing future input.
	rank := map[string]int{}
	for i, id := range n.skipped {
		rank[id] = i + 1
	}
	s.WaitQ.Reorder(func(rs []*Request) {
		sort.SliceStable(rs, func(i, j int) bool { a, c := rank[rs[i].ID], rank[rs[j].ID]; return a > 0 && (c == 0 || a < c) })
	})
	deferred, _ := s.KVCache.(DeferrableKVStore)
	blocked := map[string]bool{}
	if deferred != nil {
		for _, id := range deferred.PollDeferred(now) {
			blocked[id] = true
		}
	}
	var skipped []string
	for scan := 0; !preempted && budget > 0 && len(s.RunningBatch.Requests) < int(s.maxNumSeqs) && scan < s.WaitQ.Len(); {
		r := s.WaitQ.PeekAt(scan)
		p := n.progress(r)
		if blocked[r.ID] || p.pending > 0 && !n.store.NativeRequiresKVDelivery() {
			skipped = append(skipped, r.ID)
			scan++
			continue
		}
		if p.generation > 0 {
			r.recomputeUntil = r.InputLen() + r.EmittedTokens()
		}
		prefix := RequestCachedPrefix(s.KVCache, r)
		grant := nativeTokenGrant(r.PrefillEnd()-prefix.Tokens, budget, s.longPrefillTokenThreshold)
		if s.maxModelLen > 0 {
			grant = min(grant, max(0, s.maxModelLen-1-prefix.Tokens))
		}
		if grant <= 0 {
			panic("native waiting prefix must leave logits work")
		}
		if !n.store.ReserveNativeRange(r, prefix.Tokens, prefix.Tokens+grant, prefix.Blocks) {
			if deferred != nil && deferred.IsDeferred(r.ID) {
				skipped = append(skipped, r.ID)
				scan++
				continue
			}
			break
		}
		dequeueAdmitted(s.WaitQ, r, scan)
		s.RunningBatch.Requests = append(s.RunningBatch.Requests, r)
		r.State = StateRunning
		r.ScheduledStepIdx = s.stepCount
		s.Schedule(&ScheduledEvent{time: now, Request: r})
		s.Metrics.RequestSchedulingDelays[r.ID] = now - r.ArrivalTime
		p.scheduled = prefix.Tokens
		b.work = append(b.work, NativeWork{Request: r, Generation: p.generation, Start: prefix.Tokens, End: prefix.Tokens + grant, KnownTokens: r.InputLen() + r.EmittedTokens()})
		budget -= grant
	}
	seen := map[string]bool{}
	n.skipped = nil
	for _, id := range skipped {
		n.skipped = append(n.skipped, id)
		seen[id] = true
	}
	for _, r := range s.WaitQ.Items() {
		if !seen[r.ID] && rank[r.ID] > 0 {
			n.skipped = append(n.skipped, r.ID)
		}
	}
	for i := range b.work {
		w := &b.work[i]
		p := n.progress(w.Request)
		w.Sample = w.End >= w.KnownTokens+p.placeholders
		p.scheduled = w.End
		p.pending++
		if w.Sample {
			p.placeholders++
		}
		w.Request.NumNewTokens = int(w.End - w.Start)
		s.reqNumComputedTokens[w.Request.ID] = w.End
		b.shape = append(b.shape, NativeBatchShape{Request: w.Request.ID, ContextTokens: w.Start, QueryTokens: w.End - w.Start, PromptQueryTokens: max(0, min(w.End, w.Request.InputLen())-w.Start)})
	}
	b.kv = n.store.CaptureNativeBatch(b.work)
	s.recordQueueSnapshots()
	return b
}

func (n *nativeAsyncRuntime) consumeWork(w NativeWork, now int64, batch *nativeQueuedBatch) {
	r := w.Request
	p := n.progress(r)
	p.pending--
	if p.pending < 0 {
		panic("unmatched native batch consumption")
	}
	if r.State == StateTimedOut || r.State == StateCompleted {
		if p.pending == 0 {
			delete(n.requests, r.ID)
		}
		return
	}
	if w.Generation != p.generation && n.store.NativeRequiresKVDelivery() {
		// Pinned LMCache kv_both has requires_kv_delivery=True. Native marks
		// these outputs stale at preemption and drops them, rather than treating
		// them as observed tokens for the next generation's recomputation.
		if w.Sample {
			event := n.event(now, "native_async_stale_output_dropped", batch, nil)
			event.Request, event.Generation = r.ID, w.Generation
		}
		return
	}
	if w.Generation == p.generation {
		r.ProgressIndex = w.End
		if w.Sample {
			p.placeholders--
			if p.placeholders < 0 {
				panic("native placeholder underflow")
			}
		}
	}
	if w.Sample {
		// Offline stopping/output is consulted only here, after actual work.
		emitted := r.preservedOutputTokens
		if emitted < int64(len(r.OutputTokens)) && emitted < int64(r.MaxOutputLen) {
			if !r.TTFTSet {
				r.TTFTSet = true
				r.FirstTokenTime = now - r.ArrivalTime
				n.s.Metrics.RequestTTFTs[r.ID] = float64(r.FirstTokenTime)
			} else {
				r.ITL = append(r.ITL, now-r.lastOutputTime)
			}
			r.lastOutputTime = now
			r.preservedOutputTokens = emitted + 1
		}
	}
	finished := r.preservedOutputTokens >= int64(len(r.OutputTokens)) || r.preservedOutputTokens >= int64(r.MaxOutputLen)
	if n.s.maxModelLen > 0 && r.InputLen()+r.preservedOutputTokens >= n.s.maxModelLen {
		finished = true
		r.LengthCapped = true
		n.s.Metrics.LengthCappedRequests++
		if metric, ok := n.s.Metrics.Requests[r.ID]; ok {
			metric.LengthCapped = true
			n.s.Metrics.Requests[r.ID] = metric
		}
	}
	if !finished {
		return
	}
	r.State = StateCompleted
	n.finished = append(n.finished, r.ID)
	// Metrics count actual emitted output, not speculative scheduled positions.
	r.ProgressIndex = r.InputLen() + r.preservedOutputTokens - 1
	n.s.KVCache.ReleaseKVBlocks(r)
	r.FinishedStepIdx = n.s.stepCount
	if n.s.RunningBatch != nil {
		var keep []*Request
		for _, q := range n.s.RunningBatch.Requests {
			if q != r {
				keep = append(keep, q)
			}
		}
		n.s.RunningBatch.Requests = keep
	}
	n.s.WaitQ.Remove(r)
	n.s.Schedule(&RequestLeftEvent{time: now, Request: r})
	n.s.recordRequestCompletion(r)
	delete(n.s.reqNumComputedTokens, r.ID)
	if n.s.OnRequestDone != nil {
		for _, next := range n.s.OnRequestDone(r, now) {
			n.s.InjectArrival(next)
		}
	}
	if p.pending == 0 {
		delete(n.requests, r.ID)
	}
}
