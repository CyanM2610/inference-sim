package sim

// BatchCompleteEvent publishes computation only after its modeled duration.
// It is enabled explicitly by SimConfig.BatchCompletionEvents.
type BatchCompleteEvent struct {
	At, Start, Duration int64
	Requests            []string // Requests that actually participated in computation.
}

func (e *BatchCompleteEvent) Timestamp() int64 { return e.At }
func (e *BatchCompleteEvent) Priority() int    { return PriorityAdapterLoad }
func (e *BatchCompleteEvent) Execute(s *Simulator) {
	s.KVCache.SetClock(e.At)
	if p, ok := s.KVCache.(BatchPrefixStore); ok {
		p.CompletePrefixBatch()
	}
	// All participants may time out after submission. The engine still pays the
	// submitted duration, but cancelled results cannot advance tokens or publish
	// a first output. Keep an empty batch through the normal completion path.
	if s.RunningBatch == nil {
		s.RunningBatch = &Batch{}
	}
	s.executeBatchDuration(e.Start, e.Duration)
	s.decisionBatchReturned(e.At)
	s.KVCache.MirrorToCPU(s.RunningBatch.Requests)
	remaining := s.processCompletions(e.Start, e.Duration)
	if s.postStep == nil {
		s.scheduleNextStep(e.Start, e.Duration, remaining)
		return
	}
	// Output timestamps and completion state are committed now. During host
	// service a timeout may remove a remaining request; never restore this slice.
	s.RunningBatch.Requests = remaining
	s.finishPostStep(e.At, func(at int64) {
		var live []*Request
		if s.RunningBatch != nil {
			live = s.RunningBatch.Requests
		}
		s.scheduleNextStep(at, 0, live)
	})
}

// EventDrivenKVStore binds transfer completions to the engine's existing event queue.
type EventDrivenKVStore interface {
	BindEvents(schedule func(Event), wake func(int64))
}

// BatchWork carries the actual pre-step context, including first-admission
// prefix hits that are not yet reflected by Request.ProgressIndex.
type BatchWork struct {
	Request                 *Request
	PrefixTokens, NewTokens int64
}

// AsyncBatchExecutor lets a detailed backend execute gather/forward/scatter
// dependencies and publish the batch only at their actual completion. Returning
// false leaves the normal latency backend in control. Implementations must not
// call done before returning true; completion must arrive through an event.
type AsyncBatchExecutor interface {
	BeginBatch(now int64, work []BatchWork, done func(at int64)) (bool, error)
}

// AsyncIdleExecutor advances control-only engine steps while I/O remains.
// Empty steps must have positive duration and must stop once the backend drains.
type AsyncIdleExecutor interface {
	ExecutesEmptySteps() bool
	HasPendingEngineWork() bool
}

func pendingEngineWork(store KVStore) bool {
	b, ok := store.(AsyncIdleExecutor)
	return ok && b.ExecutesEmptySteps() && b.HasPendingEngineWork()
}

// EngineIdleCompleteEvent completes CPU control work, without reporting GPU
// computation, advancing tokens, or publishing computed cache blocks.
type EngineIdleCompleteEvent struct{ At int64 }

func (e *EngineIdleCompleteEvent) Timestamp() int64 { return e.At }
func (e *EngineIdleCompleteEvent) Priority() int    { return PriorityStep }
func (e *EngineIdleCompleteEvent) Execute(s *Simulator) {
	// Keep the in-flight marker through batch formation: reserving a new load
	// may issue a wakeup, which must not enqueue a duplicate co-timed step.
	s.finishPostStep(e.At, s.Step)
}

func (s *Simulator) beginAsyncBatch(now int64) bool {
	executor, ok := s.KVCache.(AsyncBatchExecutor)
	if !ok {
		return false
	}
	if gate, ok := s.KVCache.(interface{ AsyncBatchEnabled() bool }); ok && !gate.AsyncBatchEnabled() {
		return false
	}
	var work []BatchWork
	ev := &BatchCompleteEvent{Start: now}
	var requests []*Request
	if s.RunningBatch != nil {
		requests = s.RunningBatch.Requests
	}
	for _, r := range requests {
		if r.NumNewTokens <= 0 {
			continue
		}
		prefix := r.ProgressIndex
		if prefix < r.PrefillEnd() {
			prefix = s.reqNumComputedTokens[r.ID] - int64(r.NumNewTokens)
		}
		if prefix < 0 {
			panic("negative pre-step context")
		}
		work = append(work, BatchWork{Request: r, PrefixTokens: prefix, NewTokens: int64(r.NumNewTokens)})
		ev.Requests = append(ev.Requests, r.ID)
	}
	if len(work) == 0 {
		if !pendingEngineWork(s.KVCache) {
			return false
		}
	}
	// Keep an unscheduled in-flight marker so arrivals/transfer wakeups cannot
	// start another batch while the detailed execution graph owns this batch.
	previous := s.stepEvent
	s.stepEvent = ev
	completed := false
	returned := false
	accepted := false
	handled, err := executor.BeginBatch(now, work, func(at int64) {
		if !returned || !accepted || completed || at <= now {
			panic("invalid asynchronous batch completion")
		}
		completed = true
		ev.At = at
		ev.Duration = at - now
		if len(work) == 0 {
			s.Schedule(&EngineIdleCompleteEvent{At: at})
		} else {
			s.Schedule(ev)
		}
	})
	returned = true
	accepted = handled
	if err != nil {
		panic(err)
	}
	if !handled {
		s.stepEvent = previous
		return false
	}
	return true
}
