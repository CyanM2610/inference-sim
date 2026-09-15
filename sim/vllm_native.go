package sim

import (
	"fmt"
	"sort"
)

// VLLMNativeBatchFormation implements the synchronous dense V1 FCFS decision
// path at VLLMNativeRevision. KVStore owns allocation/transfer lifetimes. The
// simulator continues to own completion, emitted outputs, events and metrics.
// Unlike queue-only FCFS, this also fixes token grants and preemption semantics.
// Chunked prefill is enabled; priority/speculative/encoder/LoRA/PD are excluded.
type VLLMNativeBatchFormation struct {
	skipped []string
}

const VLLMNativeRevision = "79d3c28f53ea18a4a117ecb246dd6c96638041dd"

func NewVLLMNativeBatchFormation() *VLLMNativeBatchFormation {
	return &VLLMNativeBatchFormation{}
}

// SetVLLMNativeScheduler installs the complete baseline before execution. Stores
// must support observed-output recomputation; resetting TTFT on eviction would
// silently change the meaning of this baseline.
func (s *Simulator) SetVLLMNativeScheduler() error {
	if s.stepCount != 0 || s.specEnabled || s.residentAdapters != nil {
		return fmt.Errorf("vllm_native requires an idle dense non-speculative instance")
	}
	history, ok := s.KVCache.(OutputHistoryStore)
	if !ok || !history.PreservesOutputHistory() {
		return fmt.Errorf("vllm_native requires KV support for observed output history")
	}
	s.scheduler = &FCFSScheduler{}
	s.batchFormation = NewVLLMNativeBatchFormation()
	return nil
}

func (v *VLLMNativeBatchFormation) FormBatch(ctx BatchContext) BatchResult {
	if ctx.CapacityVictims != nil || len(ctx.PrefillPreemptions) != 0 || len(ctx.PreemptionStorage) != 0 || len(ctx.TokenLimits) != 0 || ctx.WaitingEligible != nil || ctx.RunningEligible != nil || ctx.DecodeTokensPerStep != nil || ctx.AdapterResident != nil {
		panic("vllm_native cannot combine with custom admission/token/preemption mechanisms")
	}
	if ctx.RunningBatch == nil {
		ctx.RunningBatch = &Batch{}
	}
	result := BatchResult{RunningBatch: ctx.RunningBatch}
	if ctx.ObserveExecutionWork {
		result.ExecutionWork = &BatchExecutionWork{TokenChecksKnown: true}
		ctx.executionWork = result.ExecutionWork
	}
	budget := ctx.MaxNumBatchedTokens
	for _, r := range ctx.RunningBatch.Requests {
		r.NumNewTokens = 0
	}
	// vLLM has no decode-first reordering: an earlier resident prefill can use
	// the budget before a later decode. Only already emitted history is visible.
	preemption := VLLMBatchFormation{preemptionPolicy: PreemptionFCFS}
	for index := 0; index < len(result.RunningBatch.Requests) && budget > 0; index++ {
		r := result.RunningBatch.Requests[index]
		demand := max(r.PrefillEnd(), r.InputLen()+r.EmittedTokens()) - r.ProgressIndex
		tokens := nativeTokenGrant(demand, budget, ctx.PrefillTokenThreshold)
		if ctx.MaxModelLen > 0 {
			tokens = min(tokens, max(0, ctx.MaxModelLen-1-r.ProgressIndex))
		}
		ctx.tokenCheck(r, "running", r.ProgressIndex, tokens)
		if tokens == 0 {
			continue
		}
		ok, _ := preemption.preemptForTokens(r, tokens, &result, ctx, &budget, index)
		if !ok {
			break
		}
		r.NumNewTokens = int(tokens)
		ctx.ComputedTokens[r.ID] = r.ProgressIndex + tokens
		budget -= tokens
	}
	// The native FCFS scheduler has a persistent skipped_waiting queue ahead
	// of ordinary waiting (including requests just preempted). Keep one owning
	// simulator WaitQ, with only IDs here, so cancellation/metrics remain shared.
	rank := map[string]int{}
	for i, id := range v.skipped {
		rank[id] = i + 1
	}
	ctx.WaitQ.Reorder(func(rs []*Request) {
		sort.SliceStable(rs, func(i, j int) bool {
			a, b := rank[rs[i].ID], rank[rs[j].ID]
			return a > 0 && (b == 0 || a < b)
		})
	})
	blocked := map[string]bool{}
	deferrable, _ := ctx.KVCache.(DeferrableKVStore)
	if deferrable != nil {
		for _, id := range deferrable.PollDeferred(ctx.Now) {
			blocked[id] = true
		}
	}
	ctx.readyRestores = map[string]bool{}
	if ready, ok := ctx.KVCache.(DeferredAdmissionStore); ok {
		for _, id := range ready.ReadyDeferredRequests() {
			ctx.readyRestores[id] = true
		}
	}
	stepSkipped := []string{}
	for scan := 0; !result.PreemptionHappened && budget > 0 && len(result.RunningBatch.Requests) < int(ctx.MaxNumSeqs) && scan < ctx.WaitQ.Len(); {
		r := ctx.WaitQ.PeekAt(scan)
		if blocked[r.ID] {
			stepSkipped = append(stepSkipped, r.ID)
			scan++
			continue
		}
		prefix := ctx.cachedPrefix(r)
		tokens := nativeTokenGrant(r.PrefillEnd()-prefix.Tokens, budget, ctx.PrefillTokenThreshold)
		if tokens <= 0 {
			panic("vllm_native prefix lookup must leave a token for logits")
		}
		ctx.tokenCheck(r, "waiting", prefix.Tokens, tokens)
		if !ctx.allocate(r, prefix.Tokens, prefix.Tokens+tokens, prefix.Blocks) {
			if deferrable != nil && deferrable.IsDeferred(r.ID) {
				stepSkipped = append(stepSkipped, r.ID)
				scan++
				continue
			}
			break // capacity failure is head-of-line, never a waiting victim
		}
		dequeueAdmitted(ctx.WaitQ, r, scan)
		result.RunningBatch.Requests = append(result.RunningBatch.Requests, r)
		result.NewlyScheduled = append(result.NewlyScheduled, ScheduledRequest{Request: r})
		r.ScheduledStepIdx, r.State, r.NumNewTokens = ctx.StepCount, StateRunning, int(tokens)
		ctx.ComputedTokens[r.ID] = prefix.Tokens + tokens
		budget -= tokens
	}
	// Native step_skipped_waiting.prepend followed by prepend_requests reverses
	// twice: visited blocked requests keep scan order ahead of unvisited skips.
	pending := map[string]bool{}
	for _, r := range ctx.WaitQ.Items() {
		pending[r.ID] = true
	}
	v.skipped = v.skipped[:0]
	for _, id := range stepSkipped {
		if pending[id] {
			v.skipped = append(v.skipped, id)
			delete(pending, id)
		}
	}
	for _, r := range ctx.WaitQ.Items() {
		if pending[r.ID] && rank[r.ID] > 0 {
			v.skipped = append(v.skipped, r.ID)
		}
	}
	// Publish the same combined queue view that the next native selection sees.
	rank = map[string]int{}
	for i, id := range v.skipped {
		rank[id] = i + 1
	}
	ctx.WaitQ.Reorder(func(rs []*Request) {
		sort.SliceStable(rs, func(i, j int) bool {
			a, b := rank[rs[i].ID], rank[rs[j].ID]
			return a > 0 && (b == 0 || a < b)
		})
	})
	return result
}

func nativeTokenGrant(demand, budget, threshold int64) int64 {
	if threshold > 0 {
		demand = min(demand, threshold)
	}
	return max(0, min(demand, budget))
}
