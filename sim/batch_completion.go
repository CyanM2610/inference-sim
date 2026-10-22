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
	s.executeBatchDuration(e.Start, e.Duration)
	s.KVCache.MirrorToCPU(s.RunningBatch.Requests)
	remaining := s.processCompletions(e.Start, e.Duration)
	s.scheduleNextStep(e.Start, e.Duration, remaining)
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
	for _, r := range s.RunningBatch.Requests {
		if r.NumNewTokens <= 0 {
			continue
		}
		prefix := r.ProgressIndex
		if prefix < r.InputLen() {
			prefix = s.reqNumComputedTokens[r.ID] - int64(r.NumNewTokens)
		}
		if prefix < 0 {
			panic("negative pre-step context")
		}
		work = append(work, BatchWork{Request: r, PrefixTokens: prefix, NewTokens: int64(r.NumNewTokens)})
		ev.Requests = append(ev.Requests, r.ID)
	}
	if len(work) == 0 {
		return false
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
		s.Schedule(ev)
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
