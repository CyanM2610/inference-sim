package sim

import "fmt"

// EnableDecisionPrefillWaits extends deadline waits to resident prefills.
// A wait retains KV and sequence ownership; it is not a preemption.
func (s *Simulator) EnableDecisionPrefillWaits() error {
	_, native := s.batchFormation.(*VLLMBatchFormation)
	if s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents || !native {
		return fmt.Errorf("prefill waits require a policy, VLLMBatchFormation and batch completion events before execution")
	}
	if err := s.enableDecisionEvents(); err != nil {
		return err
	}
	s.decision.prefillWaits = true
	return nil
}

func (s *Simulator) markDecisionPrefillWaitable(v *DecisionView) {
	if s.decision == nil || !s.decision.prefillWaits {
		return
	}
	v.Capabilities.PrefillWaits = true
	blocked := map[string]bool{}
	for _, r := range v.KV.Requests {
		blocked[r.ID] = r.TransferPending || r.Deferred
	}
	for i, r := range v.Running {
		live := s.RunningBatch.Requests[i]
		v.Running[i].PrefillWaitable = r.State == StateRunning && r.ComputedTokens > 0 && r.ComputedTokens < r.InputTokens &&
			!live.TTFTSet && !s.decision.events.firstOutput[r.ID] && !blocked[r.ID]
	}
}

func minimumRunningGrants(v DecisionView, p DecisionPlan) int64 {
	if !v.Capabilities.PrefillWaits && len(p.PrefillDeferrals) == 0 {
		return int64(len(v.Running) - len(p.Preemptions))
	}
	removed := map[string]bool{}
	for _, id := range p.Preemptions {
		removed[id] = true
	}
	for _, id := range p.PrefillDeferrals {
		removed[id] = true
	}
	updates := map[string]int64{}
	for _, w := range p.Waits {
		updates[w.Request] = w.UntilUS
	}
	var count int64
	for _, r := range v.Running {
		if removed[r.ID] {
			continue
		}
		until := r.WaitUntilUS
		if updated, ok := updates[r.ID]; ok {
			until = updated
		}
		if v.Capabilities.PrefillWaits && r.PrefillWaitable && until > v.NowUS {
			continue
		}
		count++
	}
	return count
}

func (s *Simulator) runningWaitGate(now int64) func(*Request) bool {
	if s.decision == nil || (!s.decision.prefillWaits && !s.decision.prefillDeferrals) {
		return nil
	}
	d := s.decision
	d.batchWaits = nil
	return func(r *Request) bool {
		if d.deferredPrefills[r.ID] {
			return false
		}
		until := s.decisionWaitsUntil(r.ID)
		if r.ProgressIndex < r.InputLen() && until > now {
			if d.batchWaits == nil {
				d.batchWaits = map[*Request]int64{}
			}
			d.batchWaits[r] = until
			return false
		}
		return true
	}
}

func (s *Simulator) prefillWaitsLeaveNoWork() bool {
	if s.decision == nil || (!s.decision.prefillWaits && !s.decision.prefillDeferrals) || s.RunningBatch == nil || len(s.RunningBatch.Requests) == 0 {
		return false
	}
	for _, r := range s.RunningBatch.Requests {
		if r.NumNewTokens > 0 {
			return false
		}
	}
	return true
}

// A deadline can expire while the completion service owns the in-flight
// marker. Its timer cannot schedule a second step then; resume at service end.
func (s *Simulator) resumeExpiredPrefillWait(now int64) {
	if s.decision == nil || !s.decision.prefillWaits || s.RunningBatch == nil {
		return
	}
	for _, r := range s.RunningBatch.Requests {
		if until, ok := s.decision.batchWaits[r]; ok && until <= now && r.State == StateRunning {
			s.ScheduleStepIfIdle(now)
			return
		}
	}
}

func prefillWaitUpdates(v DecisionView, p DecisionPlan) int64 {
	if !v.Capabilities.PrefillWaits || len(p.Waits) == 0 {
		return 0
	}
	running := map[string]bool{}
	for _, r := range v.Running {
		running[r.ID] = true
	}
	var count int64
	for _, w := range p.Waits {
		if running[w.Request] {
			count++
		}
	}
	return count
}
