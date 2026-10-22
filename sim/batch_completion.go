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
