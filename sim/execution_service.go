package sim

import "fmt"

// ExecutionCostModel prices completed batch-formation operations, including
// failed attempts and actual preemptions. It never receives a victim proposal
// or mutable request/cache state. The additional host service precedes compute.
type ExecutionCostModel interface {
	EstimateExecution(BatchExecutionWork, int64) (DecisionCostEstimate, error)
}

func (s *Simulator) SetExecutionCostModel(model ExecutionCostModel) error {
	if model == nil || s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents {
		return fmt.Errorf("execution service requires a policy and batch completion events before execution")
	}
	if b, ok := s.KVCache.(BatchPrefixStore); ok && b.BatchPrefixReuseEnabled() {
		return fmt.Errorf("execution service does not yet support unsubmitted prefix donor cancellation")
	}
	if s.decision.executionOutcomeCost != nil {
		return fmt.Errorf("choose one execution service model")
	}
	s.decision.executionCost = model
	s.decision.executionWork = true
	return nil
}

func cloneExecutionWork(w BatchExecutionWork) BatchExecutionWork {
	w.TokenChecks = append([]ExecutionTokenCheck(nil), w.TokenChecks...)
	w.Allocations = append([]ExecutionAllocation(nil), w.Allocations...)
	w.Prefixes = append([]ExecutionPrefix(nil), w.Prefixes...)
	for i := range w.Allocations {
		if p := w.Allocations[i].Failure; p != nil {
			copy := *p
			w.Allocations[i].Failure = &copy
		}
	}
	return w
}

// Formation currently commits allocator metadata atomically at service start.
// Resources remain owned throughout this host interval; forward and staged
// engine transfers cannot begin until it ends. Per-call CPU interleaving is
// outside this service model and must not be inferred from its end timestamp.
func (s *Simulator) prepareExecutionService(now int64, result BatchResult, control bool) {
	d := s.decision
	if d == nil || d.executionCost == nil && d.executionOutcomeCost == nil {
		return
	}
	work := BatchExecutionWork{TokenChecksKnown: control}
	if result.ExecutionWork != nil {
		work = *result.ExecutionWork
	}
	var cost DecisionCostEstimate
	var err error
	if d.executionOutcomeCost != nil {
		cost, err = d.executionOutcomeCost.EstimateExecutionOutcome(cloneDecisionView(d.record.View), cloneDecisionPlan(d.record.Plan), observedExecutionOutcome(result, work, control))
	} else {
		cost, err = d.executionCost.EstimateExecution(cloneExecutionWork(work), int64(len(result.Preempted)))
	}
	if err == nil {
		err = cost.validate(now)
	}
	if err != nil {
		panic(err)
	}
	d.record.ExecutionService = &DecisionService{StartUS: now, EndUS: now + cost.ExtraUS, ExtraUS: cost.ExtraUS, Provenance: cost.Provenance}
	if cost.ExtraUS == 0 {
		return
	}
	e := &ExecutionCompleteEvent{At: now + cost.ExtraUS, control: control, waiting: map[*Request]bool{}, preemptionsBefore: s.Metrics.PreemptionCount}
	for _, r := range s.WaitQ.Items() {
		e.waiting[r] = true
	}
	d.executionPending = e
	s.stepEvent = e
	s.Schedule(e)
}

type ExecutionCompleteEvent struct {
	At                int64
	control, revisit  bool
	preemptionsBefore int64
	waiting           map[*Request]bool
}

func (e *ExecutionCompleteEvent) Timestamp() int64 { return e.At }
func (e *ExecutionCompleteEvent) Priority() int    { return PriorityStep }
func (e *ExecutionCompleteEvent) Execute(s *Simulator) {
	if s.decision == nil || s.decision.executionPending != e {
		panic("unexpected execution completion")
	}
	s.decision.executionPending = nil
	s.KVCache.SetClock(e.At)
	s.finishExecutionFeedback(e.At)
	if e.control {
		s.continueDecisionControl(e.At)
		return
	}
	// A timeout may have removed every admitted request. It does not refund
	// completed host work, resurrect the batch, or emit a cancelled output.
	if s.RunningBatch == nil {
		s.RunningBatch = &Batch{}
	}
	s.executeScheduledBatch(e.At, e.preemptionsBefore)
	late := e.revisit
	for _, r := range s.WaitQ.Items() {
		late = late || !e.waiting[r]
	}
	if late {
		s.ScheduleStepIfIdle(e.At)
	}
}
