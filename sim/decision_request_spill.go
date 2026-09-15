package sim

import "fmt"

// DecisionSpillTarget is a detached current-state inventory. AvailableBlocks
// excludes this prefix's existing copies and includes other evictable copies;
// the pool policy may still decline eviction. It contains no completion times.
type DecisionSpillTarget struct {
	Pool            string `json:"pool"`
	CompleteBlocks  int64  `json:"complete_blocks"`
	ReadyBlocks     int64  `json:"ready_blocks"`
	PendingBlocks   int64  `json:"pending_blocks"`
	MissingBlocks   int64  `json:"missing_blocks"`
	AvailableBlocks int64  `json:"available_blocks"`
	TailTokens      int64  `json:"tail_tokens"`
}

// DecisionPreemptionStorage applies only if this step actually preempts Request.
// Recompute creates no new copy; it does not delete shared hits or cancel DMA.
type DecisionPreemptionStorage struct {
	Request string `json:"request"`
	Mode    string `json:"mode"`
	Pool    string `json:"pool,omitempty"`
}

// RequestSpillOutcome describes accepted copies, not a residency guarantee.
// Pending blocks are never reported as ready. TailTokens need recomputation.
type RequestSpillOutcome struct {
	Request       string `json:"request"`
	Pool          string `json:"pool,omitempty"`
	Status        string `json:"status"`
	ReadyBlocks   int64  `json:"ready_blocks"`
	PendingBlocks int64  `json:"pending_blocks"`
	NewBlocks     int64  `json:"new_blocks"`
	TailTokens    int64  `json:"tail_tokens"`
}

// RequestSpillStore owns all destination reservations and source lifetimes.
// SpillPrefill reserves the complete missing prefix before submitting copies;
// a capacity failure returns recompute_capacity with no new transfer/reservation.
type RequestSpillStore interface {
	SupportsRequestSpill() bool
	SpillPrefill(*Request, string) RequestSpillOutcome
}

func (s *Simulator) EnableDecisionRequestSpill() error {
	if s.decision == nil || !s.decision.prefillPreemption || s.stepCount != 0 {
		return fmt.Errorf("request spill requires prefill preemption before execution")
	}
	backend, ok := s.KVCache.(RequestSpillStore)
	if !ok || !backend.SupportsRequestSpill() {
		return fmt.Errorf("backend does not support request spill")
	}
	s.decision.requestSpill = true
	return nil
}

func validateDecisionPreemptionStorage(v DecisionView, p DecisionPlan) error {
	if len(p.PreemptionStorage) == 0 {
		return nil
	}
	if !v.Capabilities.RequestSpill {
		return fmt.Errorf("request spill is unsupported")
	}
	victims := map[string]bool{}
	for _, id := range p.Preemptions {
		victims[id] = true
	}
	if p.CapacityVictims != nil {
		for _, id := range p.CapacityVictims.Order {
			victims[id] = true
		}
	}
	pools := map[string]bool{}
	for _, pool := range v.KV.Pools {
		pools[pool.ID] = true
	}
	seen := map[string]bool{}
	for _, action := range p.PreemptionStorage {
		if !victims[action.Request] || seen[action.Request] {
			return fmt.Errorf("storage action requires a unique preemption victim %q", action.Request)
		}
		seen[action.Request] = true
		switch action.Mode {
		case "spill":
			if action.Pool == "" || !pools[action.Pool] {
				return fmt.Errorf("spill target is not accessible: %q", action.Pool)
			}
		case "recompute":
			if action.Pool != "" {
				return fmt.Errorf("recompute has no spill target")
			}
		default:
			return fmt.Errorf("unknown preemption storage mode %q", action.Mode)
		}
	}
	return nil
}

// Called before progress/reset/release at either explicit or capacity preemption.
func preparePreemptionStorage(req *Request, result *BatchResult, ctx BatchContext) {
	for _, action := range ctx.PreemptionStorage {
		if action.Request != req.ID {
			continue
		}
		var outcome RequestSpillOutcome
		switch action.Mode {
		case "spill":
			backend, ok := ctx.KVCache.(RequestSpillStore)
			if !ok || !backend.SupportsRequestSpill() {
				panic("request spill reached an unsupported backend")
			}
			outcome = backend.SpillPrefill(req, action.Pool)
		case "recompute":
			outcome = RequestSpillOutcome{Request: req.ID, Status: "recompute", TailTokens: req.ProgressIndex}
		default:
			panic("unvalidated preemption storage mode")
		}
		result.PreemptionStorage = append(result.PreemptionStorage, outcome)
		break
	}
	if store, ok := ctx.KVCache.(PrefillPreemptionStore); ok {
		store.PreparePrefillPreemption(req)
	}
}
