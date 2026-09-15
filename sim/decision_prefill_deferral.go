package sim

import "fmt"

// EnableDecisionPrefillDeferrals enables per-round resident prefill selection.
// Deadline waits are a separate capability and retain their own lifetime.
func (s *Simulator) EnableDecisionPrefillDeferrals() error {
	_, native := s.batchFormation.(*VLLMBatchFormation)
	if s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents || !native {
		return fmt.Errorf("prefill deferrals require a policy, VLLMBatchFormation and batch completion events before execution")
	}
	if err := s.enableDecisionEvents(); err != nil {
		return err
	}
	s.decision.prefillDeferrals = true
	return nil
}

func (s *Simulator) markDecisionPrefillDeferrable(v *DecisionView) {
	if s.decision == nil || !s.decision.prefillDeferrals {
		return
	}
	v.Capabilities.PrefillDeferrals = true
	blocked := map[string]bool{}
	for _, r := range v.KV.Requests {
		blocked[r.ID] = r.TransferPending || r.Deferred
	}
	for i, r := range v.Running {
		live := s.RunningBatch.Requests[i]
		v.Running[i].PrefillDeferrable = r.State == StateRunning && r.ComputedTokens > 0 && r.ComputedTokens < r.InputTokens &&
			r.EmittedTokens == 0 && !live.TTFTSet && !s.decision.events.firstOutput[r.ID] && !blocked[r.ID]
	}
}

func validateDecisionPrefillDeferrals(v DecisionView, p DecisionPlan) error {
	if len(p.PrefillDeferrals) == 0 {
		return nil
	}
	if !v.Capabilities.PrefillDeferrals {
		return fmt.Errorf("resident prefill deferrals are unsupported")
	}
	eligible, seen, preempted := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, r := range v.Running {
		eligible[r.ID] = r.PrefillDeferrable && r.State == StateRunning && r.ComputedTokens > 0 && r.ComputedTokens < r.InputTokens && r.EmittedTokens == 0
	}
	for _, id := range p.Preemptions {
		preempted[id] = true
	}
	for _, id := range p.PrefillDeferrals {
		if !eligible[id] || seen[id] || preempted[id] {
			return fmt.Errorf("invalid, repeated or preempted prefill deferral %q", id)
		}
		seen[id] = true
	}
	if minimumRunningGrants(v, p) > 0 {
		return nil
	}
	// Allow attempts to admit waiting work, without promising allocation. The
	// existing allocation/transfer runtime supplies actual blocking outcomes.
	if p.Admission == nil && len(v.Waiting) > 0 || p.Admission != nil && len(p.Admission.Requests) > 0 {
		return nil
	}
	if v.KV.TransfersKnown && len(v.KV.PendingTransfers) > 0 {
		return nil
	}
	revisit := v.RevisitAtUS
	if p.RevisitAtUS != nil {
		revisit = *p.RevisitAtUS
	}
	if v.Capabilities.Revisit && revisit > v.NowUS {
		return nil
	}
	updates := map[string]int64{}
	for _, w := range p.Waits {
		updates[w.Request] = w.UntilUS
	}
	if v.Capabilities.WaitUntil {
		for _, group := range [][]DecisionRequest{v.Waiting, v.Running} {
			for _, r := range group {
				until := r.WaitUntilUS
				if updated, ok := updates[r.ID]; ok {
					until = updated
				}
				if until > v.NowUS {
					return nil
				}
			}
		}
	}
	return fmt.Errorf("prefill deferrals leave no runnable work or declared wakeup")
}
