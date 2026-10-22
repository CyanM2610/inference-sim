package sim

import (
	"container/heap"
	"sort"
)

type decisionWaitState struct {
	request *Request
	until   int64
}

func (s *Simulator) decisionWaitsUntil(id string) int64 {
	if s.decision == nil {
		return 0
	}
	return s.decision.waits[id].until
}

func waitOutcome(id string, until, now int64) DecisionWaitOutcome {
	status := "waiting"
	if until == 0 {
		status = "cleared"
	} else if until <= now {
		status = "elapsed"
	}
	return DecisionWaitOutcome{Request: id, UntilUS: until, Status: status}
}

func (s *Simulator) applyDecisionWaits(now int64) {
	d := s.decision
	p := d.record.Plan
	if len(p.Waits) == 0 && p.RevisitAtUS == nil {
		return
	}
	if len(p.Waits) > 0 {
		if d.waits == nil {
			d.waits = map[string]decisionWaitState{}
		}
		waiting := map[string]*Request{}
		for _, r := range s.WaitQ.Items() {
			waiting[r.ID] = r
		}
		if d.prefillWaits && s.RunningBatch != nil {
			for _, r := range s.RunningBatch.Requests {
				waiting[r.ID] = r
			}
		}
		for _, wait := range p.Waits {
			result := waitOutcome(wait.Request, wait.UntilUS, now)
			d.record.Feedback.WaitUpdates = append(d.record.Feedback.WaitUpdates, result)
			delete(d.waits, wait.Request)
			if result.Status == "waiting" {
				d.waits[wait.Request] = decisionWaitState{request: waiting[wait.Request], until: wait.UntilUS}
			}
		}
	}
	if p.RevisitAtUS != nil {
		result := waitOutcome("", *p.RevisitAtUS, now)
		d.record.Feedback.Revisit = &result
		d.revisitAt = 0
		if result.Status == "waiting" {
			d.revisitAt = *p.RevisitAtUS
		}
	}
	s.refreshDecisionTimer(now)
}

// Only the nearest deadline occupies the event queue. Remove cancelled/replaced
// events physically: a lazy no-op must not advance the cluster's clock.
func (s *Simulator) refreshDecisionTimer(now int64) {
	d := s.decision
	if d == nil || (d.timer == nil && len(d.waits) == 0 && d.revisitAt == 0) {
		return
	}
	waiting := map[*Request]bool{}
	live := false
	for _, r := range s.WaitQ.Items() {
		if r.State != StateCompleted && r.State != StateTimedOut {
			waiting[r] = true
			live = true
		}
	}
	if s.RunningBatch != nil {
		for _, r := range s.RunningBatch.Requests {
			if r.State != StateCompleted && r.State != StateTimedOut {
				live = true
				if d.prefillWaits {
					waiting[r] = true
				}
			}
		}
	}
	if !live {
		d.revisitAt = 0
	}
	next := d.revisitAt
	for id, wait := range d.waits {
		if !waiting[wait.request] {
			delete(d.waits, id)
			continue
		}
		if next == 0 || wait.until < next {
			next = wait.until
		}
	}
	if next > 0 {
		next = max(now, next)
	}
	if d.timer != nil && d.timer.At == next {
		return
	}
	if d.timer != nil {
		for i, entry := range s.eventQueue {
			if entry.event == d.timer {
				heap.Remove(&s.eventQueue, i)
				break
			}
		}
		d.timer = nil
	}
	if next > 0 {
		d.timer = &DecisionTimerEvent{At: next}
		s.Schedule(d.timer)
	}
}

type DecisionTimerEvent struct{ At int64 }

func (e *DecisionTimerEvent) Timestamp() int64 { return e.At }
func (e *DecisionTimerEvent) Priority() int    { return PriorityAdapterLoad }
func (e *DecisionTimerEvent) Execute(s *Simulator) {
	d := s.decision
	if d == nil || d.timer != e {
		panic("unexpected decision timer")
	}
	d.timer = nil
	var due []string
	for id, wait := range d.waits {
		if wait.until <= e.At {
			due = append(due, id)
		}
	}
	sort.Strings(due)
	for _, id := range due {
		delete(d.waits, id)
		s.notifyDecisionEvent(DecisionEvent{Kind: "timer", OccurredUS: e.At, Request: id, Reason: "wait_expired"})
	}
	if d.revisitAt > 0 && d.revisitAt <= e.At {
		d.revisitAt = 0
		s.notifyDecisionEvent(DecisionEvent{Kind: "timer", OccurredUS: e.At, Reason: "revisit"})
	}
	s.refreshDecisionTimer(e.At)
	s.ScheduleStepIfIdle(e.At)
}
