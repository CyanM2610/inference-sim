package sim

import "fmt"

// ExecutionOutcome contains facts from the batch just formed. No Request or
// cache pointers, future outputs, or estimated token grants cross this seam.
type ExecutionOutcome struct {
	Work              BatchExecutionWork
	Grants            []DecisionGrant
	Capacity          []DecisionCapacityOutcome
	Preemptions       []DecisionPreemptionOutcome
	PreemptionStorage []RequestSpillOutcome
	PreemptionCount   int64
	ControlStep       bool
}

// ExecutionOutcomeCostModel can price dispatch, actual feedback and worker
// bookkeeping together. Its service ends before Observe and before compute.
// This is an aggregate CPU interval, not a per-call interleaving model.
type ExecutionOutcomeCostModel interface {
	EstimateExecutionOutcome(DecisionView, DecisionPlan, ExecutionOutcome) (DecisionCostEstimate, error)
}

func (s *Simulator) SetExecutionOutcomeCostModel(model ExecutionOutcomeCostModel) error {
	if model == nil || s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents {
		return fmt.Errorf("execution outcome service requires a policy and completion events before execution")
	}
	if s.decision.executionCost != nil || s.decision.executionOutcomeCost != nil {
		return fmt.Errorf("choose one execution service model")
	}
	if b, ok := s.KVCache.(BatchPrefixStore); ok && b.BatchPrefixReuseEnabled() {
		return fmt.Errorf("execution outcome service does not support unsubmitted prefix donor cancellation")
	}
	s.decision.executionOutcomeCost = model
	s.decision.executionWork = true
	return nil
}

func observedExecutionOutcome(result BatchResult, work BatchExecutionWork, control bool) ExecutionOutcome {
	out := ExecutionOutcome{Work: cloneExecutionWork(work), ControlStep: control,
		Capacity: append([]DecisionCapacityOutcome(nil), result.Capacity...), PreemptionCount: int64(len(result.Preempted)),
		PreemptionStorage: append([]RequestSpillOutcome(nil), result.PreemptionStorage...)}
	if result.RunningBatch != nil {
		for _, r := range result.RunningBatch.Requests {
			if r.NumNewTokens > 0 {
				out.Grants = append(out.Grants, DecisionGrant{Request: r.ID, Tokens: int64(r.NumNewTokens)})
			}
		}
	}
	for _, r := range result.Preempted {
		if r.Reason == "policy_prefill" || r.Reason == "policy_capacity_prefill" {
			out.Preemptions = append(out.Preemptions, DecisionPreemptionOutcome{Request: r.Request.ID, ComputedTokensBefore: r.ComputedTokensBefore, Status: "requeued"})
		}
	}
	return out
}

func (s *Simulator) deferExecutionFeedback(f DecisionFeedback) bool {
	d := s.decision
	if d.executionOutcomeCost == nil || d.executionPending == nil || f.Status != "applied" {
		return false
	}
	if d.executionFeedback != nil {
		panic("execution feedback already pending")
	}
	f = cloneDecisionFeedback(f)
	d.executionFeedback = &f
	return true
}

func (s *Simulator) finishExecutionFeedback(now int64) {
	d := s.decision
	if d.executionOutcomeCost == nil {
		return
	}
	if d.executionFeedback == nil {
		panic("execution service completed without its feedback")
	}
	f := *d.executionFeedback
	d.executionFeedback = nil
	s.deliverDecisionAt(f, now)
	s.flushDecisionEvents(now)
}
