package sim

import "fmt"

// DecisionCapacityReservation checks one request against present client-output
// reservations. Fits is not an admission promise: another new request, I/O, or
// protected pages may still block execution. This snapshot never reserves space.
type DecisionCapacityReservation struct {
	NeededBlocks int64 `json:"needed_blocks"`
	FreeBlocks   int64 `json:"free_blocks"`
	Fits         bool  `json:"fits"`
}

func (s *Simulator) EnableDecisionCapacityReservationView() error {
	if s.decision == nil || !s.decision.capacityPreemption || s.stepCount != 0 {
		return fmt.Errorf("capacity reservation view requires capacity preemption before execution")
	}
	backend, ok := s.KVCache.(interface{ SupportsCapacityReservationView() bool })
	if !ok || !backend.SupportsCapacityReservationView() {
		return fmt.Errorf("backend has no supported capacity reservation view")
	}
	s.decision.capacityReservationView = true
	return nil
}

// DecisionCapacityVictims is a conditional preference, not an instruction to
// evict immediately. A present empty list protects every running request.
type DecisionCapacityVictims struct {
	Order     []string                    `json:"order"`
	Admission []DecisionCapacityAdmission `json:"admission,omitempty"`
}

// A waiting request may reclaim only the named subset. An empty subset waits
// for running work rather than repeatedly displacing more urgent requests.
type DecisionCapacityAdmission struct {
	Request string   `json:"request"`
	Victims []string `json:"victims"`
}

type DecisionCapacityOutcome struct {
	Failure AllocationFailure `json:"failure"`
	Victim  string            `json:"victim,omitempty"`
	Status  string            `json:"status"`
}

func (s *Simulator) EnableDecisionCapacityPreemption() error {
	if s.decision == nil || !s.decision.prefillPreemption || s.stepCount != 0 {
		return fmt.Errorf("capacity preemption requires prefill preemption before execution")
	}
	if _, ok := s.KVCache.(AllocationFailureStore); !ok {
		return fmt.Errorf("capacity preemption requires explicit allocation failure causes")
	}
	if b, ok := s.KVCache.(interface{ DecodeCapacityReservationEnabled() bool }); !ok || !b.DecodeCapacityReservationEnabled() {
		return fmt.Errorf("decode protection requires declared output capacity reservations")
	}
	if b, ok := s.KVCache.(BatchPrefixStore); ok && b.BatchPrefixReuseEnabled() {
		return fmt.Errorf("capacity preemption cannot revoke batch prefix donors yet")
	}
	s.decision.capacityPreemption = true
	return nil
}

func validateDecisionCapacity(v DecisionView, p DecisionPlan) error {
	if p.CapacityVictims == nil {
		return nil
	}
	if !v.Capabilities.CapacityPreemption || !v.Capabilities.PrefillPreemption {
		return fmt.Errorf("capacity preemption is unsupported")
	}
	if len(p.Preemptions) > 0 {
		return fmt.Errorf("choose explicit or capacity-triggered preemption in one plan")
	}
	eligible := map[string]bool{}
	for _, r := range v.Running {
		eligible[r.ID] = r.PrefillPreemptible && r.State == StateRunning && r.ComputedTokens > 0 && r.ComputedTokens < r.InputTokens && r.EmittedTokens == 0
	}
	seen := map[string]bool{}
	for _, id := range p.CapacityVictims.Order {
		if !eligible[id] || seen[id] {
			return fmt.Errorf("invalid or repeated capacity victim %q", id)
		}
		seen[id] = true
	}
	waiting := map[string]bool{}
	for _, r := range v.Waiting {
		waiting[r.ID] = true
	}
	guarded := map[string]bool{}
	for _, a := range p.CapacityVictims.Admission {
		if !waiting[a.Request] || guarded[a.Request] {
			return fmt.Errorf("invalid capacity admission guard %q", a.Request)
		}
		guarded[a.Request] = true
		used := map[string]bool{}
		for _, id := range a.Victims {
			if !seen[id] || used[id] {
				return fmt.Errorf("invalid admission victim %q", id)
			}
			used[id] = true
		}
	}
	return nil
}

// evictCapacityVictim executes at most one conditional victim after an actual
// failed allocation. Repeated calls re-read physical allocator outcomes; they
// never assume that releasing logical tokens frees shared/pinned HBM blocks.
func evictCapacityVictim(req *Request, result *BatchResult, ctx BatchContext, tokenBudget *int64) int {
	backend, ok := ctx.KVCache.(AllocationFailureStore)
	if ctx.CapacityVictims == nil || !ok {
		panic("capacity victim invoked without supported execution context")
	}
	failure := backend.LastAllocationFailure()
	if failure.Request != req.ID || failure.Kind == "" {
		panic("allocator did not report the failed request")
	}
	outcome := DecisionCapacityOutcome{Failure: failure, Status: "waiting"}
	if failure.Kind != "capacity" {
		result.Capacity = append(result.Capacity, outcome)
		return -1
	}
	outcome.Status = "no_eligible_victim"
	var allowed map[string]bool
	for _, a := range ctx.CapacityVictims.Admission {
		if a.Request == req.ID {
			allowed = map[string]bool{}
			for _, id := range a.Victims {
				allowed[id] = true
			}
			break
		}
	}
	for _, id := range ctx.CapacityVictims.Order {
		if allowed != nil && !allowed[id] {
			continue
		}
		for index, victim := range result.RunningBatch.Requests {
			if victim.ID != id {
				continue
			}
			if victim.State != StateRunning || victim.TTFTSet || victim.ProgressIndex <= 0 || victim.ProgressIndex >= victim.InputLen() {
				panic("capacity victim no longer a preemptible prefill")
			}
			outcome.Victim, outcome.Status = id, "requeued"
			result.Capacity = append(result.Capacity, outcome)
			result.Preempted = append(result.Preempted, PreemptedRequest{Request: victim, Reason: "policy_capacity_prefill", ComputedTokensBefore: victim.ProgressIndex})
			result.RunningBatch.Requests = append(result.RunningBatch.Requests[:index], result.RunningBatch.Requests[index+1:]...)
			if victim.NumNewTokens > 0 {
				*tokenBudget += int64(victim.NumNewTokens)
			}
			preparePreemptionStorage(victim, result, ctx)
			resetPreemptedRequest(victim, ctx)
			ctx.capacityYielded[victim.ID] = true
			ctx.WaitQ.PrependFront(victim)
			return index
		}
	}
	result.Capacity = append(result.Capacity, outcome)
	return -1
}
