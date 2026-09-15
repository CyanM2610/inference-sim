package sim

import "fmt"

// DecisionPreemptionOutcome reports logical progress relinquished, not the
// amount of immediately free HBM: shared references and copy leases may remain.
type DecisionPreemptionOutcome struct {
	Request              string `json:"request"`
	ComputedTokensBefore int64  `json:"computed_tokens_before"`
	Status               string `json:"status"`
}

// PrefillPreemptionStore attaches execution dependencies specific to yielding
// a request (for example native offloader store flush), before refs are freed.
type PrefillPreemptionStore interface{ PreparePrefillPreemption(*Request) }

// EnableDecisionPrefillPreemption opts into a new action, keeping existing
// views and plans unchanged by default. Event tracking remembers the first
// output even if legacy capacity preemption later resets Request.TTFTSet.
func (s *Simulator) EnableDecisionPrefillPreemption() error {
	if s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents {
		return fmt.Errorf("prefill preemption requires a policy and batch completion events before execution")
	}
	if err := s.enableDecisionEvents(); err != nil {
		return err
	}
	s.decision.prefillPreemption = true
	return nil
}

func (s *Simulator) markDecisionPrefillPreemptible(v *DecisionView) {
	if s.decision == nil || !s.decision.prefillPreemption {
		return
	}
	v.Capabilities.PrefillPreemption = true
	v.Capabilities.RequestSpill = s.decision.requestSpill
	v.Capabilities.CapacityPreemption = s.decision.capacityPreemption
	blocked := map[string]bool{}
	for _, r := range v.KV.Requests {
		blocked[r.ID] = r.TransferPending || r.Deferred
	}
	for i, r := range v.Running {
		live := s.RunningBatch.Requests[i]
		v.Running[i].PrefillPreemptible = r.State == StateRunning && r.ComputedTokens > 0 && r.ComputedTokens < r.InputTokens &&
			!live.TTFTSet && !s.decision.events.firstOutput[r.ID] && !blocked[r.ID]
	}
}

func validateDecisionPreemptions(v DecisionView, p DecisionPlan) error {
	if len(p.Preemptions) == 0 {
		return nil
	}
	if !v.Capabilities.PrefillPreemption {
		return fmt.Errorf("prefill preemption is unsupported")
	}
	eligible := map[string]bool{}
	for _, r := range v.Running {
		eligible[r.ID] = r.PrefillPreemptible && r.State == StateRunning && r.ComputedTokens > 0 && r.ComputedTokens < r.InputTokens && r.EmittedTokens == 0
	}
	seen := map[string]bool{}
	for _, id := range p.Preemptions {
		if !eligible[id] || seen[id] {
			return fmt.Errorf("invalid or repeated prefill preemption %q", id)
		}
		seen[id] = true
	}
	for _, cap := range p.TokenCaps {
		if seen[cap.Request] {
			return fmt.Errorf("preempted request %q cannot receive a token cap in the same step", cap.Request)
		}
	}
	for _, wait := range p.Waits {
		if seen[wait.Request] {
			return fmt.Errorf("preempted request %q cannot receive a wait in the same step", wait.Request)
		}
	}
	return nil
}

// This helper is shared by capacity and explicit preemption. Physical content
// and in-flight source protection remain owned by KVStore.ReleaseKVBlocks.
func resetPreemptedRequest(req *Request, ctx BatchContext) {
	preserve := false
	if backend, ok := ctx.KVCache.(OutputHistoryStore); ok && backend.PreservesOutputHistory() {
		req.preserveOutputForRecompute()
		preserve = req.TTFTSet
	}
	req.State = StateQueued
	req.ProgressIndex = 0
	req.NumNewTokens = 0
	if !preserve {
		req.ITL = nil
		req.TTFTSet = false
	}
	req.specDecodeCarry = 0
	ctx.KVCache.ReleaseKVBlocks(req)
	delete(ctx.ComputedTokens, req.ID)
}

func applyPrefillPreemptions(result *BatchResult, ctx BatchContext) {
	// Validate the complete trusted batch input before any release or requeue.
	running := map[*Request]bool{}
	for _, req := range result.RunningBatch.Requests {
		running[req] = true
	}
	victims := map[*Request]bool{}
	for _, req := range ctx.PrefillPreemptions {
		if req == nil || !running[req] || victims[req] || req.State != StateRunning || req.TTFTSet || req.ProgressIndex <= 0 || req.ProgressIndex >= req.InputLen() {
			panic("invalid runtime prefill preemption")
		}
		victims[req] = true
	}
	kept := result.RunningBatch.Requests[:0]
	for _, req := range result.RunningBatch.Requests {
		if !victims[req] {
			kept = append(kept, req)
		}
	}
	result.RunningBatch.Requests = kept
	for _, req := range ctx.PrefillPreemptions {
		result.Preempted = append(result.Preempted, PreemptedRequest{Request: req, Reason: "policy_prefill", ComputedTokensBefore: req.ProgressIndex})
		preparePreemptionStorage(req, result, ctx)
		resetPreemptedRequest(req, ctx)
	}
	for i := len(ctx.PrefillPreemptions) - 1; i >= 0; i-- {
		ctx.WaitQ.PrependFront(ctx.PrefillPreemptions[i])
	}
	// Do not set PreemptionHappened: that capacity-failure flag stops all waiting
	// admissions. An explicit plan may admit its replacement in this very step.
}
