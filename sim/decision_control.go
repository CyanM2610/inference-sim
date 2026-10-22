package sim

import "fmt"

// EnableDecisionControlSteps includes no-request I/O and completion-cleanup
// iterations in the same policy/service boundary. It is opt-in because a
// stateful policy can react to additional Decide/Observe calls even at zero cost.
func (s *Simulator) EnableDecisionControlSteps() error {
	backend, ok := s.KVCache.(AsyncIdleExecutor)
	if s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents || !ok || !backend.ExecutesEmptySteps() {
		return fmt.Errorf("control decisions require a policy and an empty-step backend before execution")
	}
	s.decision.controlSteps = true
	return nil
}

func (s *Simulator) pendingDecisionControl() bool {
	return s.decision != nil && s.decision.controlSteps && s.decision.controlPending
}

// Completion notification is an observed host obligation, not an extra GPU
// operation. A following ordinary decision can consume it without an empty one.
func (s *Simulator) noteDecisionControlCompletion(now int64) {
	if s.decision == nil || !s.decision.controlSteps {
		return
	}
	s.decision.controlPending = true
	if s.stepEvent == nil {
		e := &StepEvent{time: now}
		s.stepEvent = e
		s.Schedule(e)
	}
}

func (s *Simulator) beginDecisionControl(now int64) {
	s.stepCount++
	s.KVCache.SetClock(now)
	s.prepareDecisionKind(now, true)
	if s.decision.pending == nil {
		s.finishDecisionControl(now)
	}
}

func (s *Simulator) finishDecisionControl(now int64) {
	d := s.decision
	d.applying = true
	s.prepareExecutionService(now, BatchResult{}, true)
	s.deliverDecision(DecisionFeedback{Version: d.record.View.Version, Status: "applied", Reason: "control_step"})
	d.applying = false
	s.flushDecisionEvents(now)
	if d.executionPending != nil {
		return
	}
	s.continueDecisionControl(now)
}

func (s *Simulator) continueDecisionControl(now int64) {
	// Requests arriving during this service were not part of its empty view.
	// Advance only the existing I/O, then let them receive a fresh decision.
	if pendingEngineWork(s.KVCache) && s.beginAsyncBatch(now) {
		return
	}
	s.completeHostService(now, nil, func(at int64) { s.finishPostStep(at, s.finishIdleDecisionControl) })
}

func (s *Simulator) finishIdleDecisionControl(now int64) {
	s.stepEvent = nil
	if s.WaitQ.Len() > 0 || s.pendingDecisionControl() || s.pendingRegistrations() {
		e := &StepEvent{time: now}
		s.stepEvent = e
		s.Schedule(e)
	}
}
