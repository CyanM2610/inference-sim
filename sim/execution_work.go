package sim

import "fmt"

// BatchExecutionWork records actual calls made by batch formation. These are
// simulator operations, not predictions or assumed native callback counts.
// In particular, a false allocation may mean asynchronous loading, not failure
// of the native allocator. Keep the observed reason for cross-backend audits.
type BatchExecutionWork struct {
	TokenChecksKnown bool                  `json:"token_checks_known,omitempty"`
	TokenChecks      []ExecutionTokenCheck `json:"token_checks,omitempty"`
	Allocations      []ExecutionAllocation `json:"allocations"`
	Prefixes         []ExecutionPrefix     `json:"prefix_queries"`
}

// ExecutionTokenCheck records one actual running visit or waiting token-limit
// check. It is not a native reserve count: async loading can bypass native
// reserve while taking this simulator path. AllocationStart and the next
// check's start (or len(Allocations)) delimit its subsequent allocation attempts.
type ExecutionTokenCheck struct {
	Request         string `json:"request"`
	Phase           string `json:"phase"`
	ComputedTokens  int64  `json:"computed_tokens"`
	Tokens          int64  `json:"tokens"`
	Held            bool   `json:"held,omitempty"`
	AllocationStart int    `json:"allocation_start"`
}

func (ctx BatchContext) tokenCheck(req *Request, phase string, computed, tokens int64) *ExecutionTokenCheck {
	w := ctx.executionWork
	if w == nil {
		return nil
	}
	w.TokenChecks = append(w.TokenChecks, ExecutionTokenCheck{Request: req.ID, Phase: phase,
		ComputedTokens: computed, Tokens: tokens, AllocationStart: len(w.Allocations)})
	return &w.TokenChecks[len(w.TokenChecks)-1]
}

type ExecutionAllocation struct {
	Request      string                  `json:"request"`
	Start        int64                   `json:"start_tokens"`
	End          int64                   `json:"end_tokens"`
	Granted      bool                    `json:"compute_admitted"`
	Failure      *AllocationFailure      `json:"failure,omitempty"`
	Dependencies []KVExecutionDependency `json:"execution_dependencies,omitempty"`
}

type ExecutionPrefix struct {
	Request         string `json:"request"`
	Tokens          int64  `json:"tokens"`
	Blocks          int64  `json:"blocks"`
	AdoptingRestore bool   `json:"adopting_restore"`
}

func (s *Simulator) EnableExecutionWorkObservation() error {
	if s.decision == nil || s.stepCount != 0 {
		return fmt.Errorf("execution observation requires a decision policy before execution")
	}
	s.decision.executionWork = true
	return nil
}

func (ctx BatchContext) allocate(req *Request, start, end int64, cached []int64) bool {
	ok := ctx.KVCache.AllocateKVBlocks(req, start, end, cached)
	if w := ctx.executionWork; w != nil {
		call := ExecutionAllocation{Request: req.ID, Start: start, End: end, Granted: ok}
		if store, known := ctx.KVCache.(KVExecutionDependencyStore); known {
			call.Dependencies = store.ExecutionDependencies(req.ID)
		}
		if !ok {
			if store, known := ctx.KVCache.(AllocationFailureStore); known {
				failure := store.LastAllocationFailure()
				call.Failure = &failure
			}
		}
		w.Allocations = append(w.Allocations, call)
	}
	return ok
}

func (ctx BatchContext) cachedPrefix(req *Request) CachedPrefix {
	p := RequestCachedPrefix(ctx.KVCache, req)
	if w := ctx.executionWork; w != nil {
		w.Prefixes = append(w.Prefixes, ExecutionPrefix{Request: req.ID, Tokens: p.Tokens, Blocks: int64(len(p.Blocks)), AdoptingRestore: ctx.readyRestores[req.ID]})
	}
	return p
}
