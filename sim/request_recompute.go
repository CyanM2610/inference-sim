package sim

// OutputHistoryStore opts into restoring prompt plus already-observed output.
// Legacy stores keep their existing request reset until their lookup/allocation
// paths support that history. This capability is an execution mechanism, not a
// policy promise and never permits reading future output tokens.
type OutputHistoryStore interface {
	PreservesOutputHistory() bool
}

// EmittedTokens survives KV eviction. ProgressIndex remains the number of KV
// positions computed, so it can temporarily fall behind the observed output.
func (r *Request) EmittedTokens() int64 {
	n := r.preservedOutputTokens
	if r.TTFTSet {
		n = max(n, r.ProgressIndex-r.InputLen()+1)
	}
	return n
}

// PrefillEnd includes the observed token history that needs recomputation.
// InputLen continues to mean the original user prompt, including for policies.
func (r *Request) PrefillEnd() int64 { return max(r.InputLen(), r.recomputeUntil) }

func (r *Request) PrefillTokens() []TokenID {
	if r.recomputeUntil <= r.InputLen() {
		return r.FullInputTokens()
	}
	n := r.recomputeUntil - r.InputLen()
	if n > r.preservedOutputTokens || n > int64(len(r.OutputTokens)) {
		panic("recompute exceeded observed output history")
	}
	tokens := make([]TokenID, 0, r.recomputeUntil)
	tokens = append(tokens, r.FullInputTokens()...)
	return append(tokens, r.OutputTokens[:n]...)
}

func (r *Request) PrefillTokenSlice(start, end int64) []TokenID {
	if end <= r.InputLen() {
		return r.InputTokenSlice(start, end)
	}
	if start < 0 || end < start || end > r.PrefillEnd() {
		panic("recompute token range outside known history")
	}
	tokens := r.PrefillTokens()
	return tokens[start:end:end]
}

func (r *Request) preserveOutputForRecompute() {
	if !r.TTFTSet {
		return
	}
	r.preservedOutputTokens = r.EmittedTokens()
	r.recomputeUntil = r.InputLen() + r.preservedOutputTokens
	// Requests built by older callers may not carry the engine timestamp yet.
	if r.lastOutputTime == 0 {
		r.lastOutputTime = r.ArrivalTime + r.FirstTokenTime
		for _, interval := range r.ITL {
			r.lastOutputTime += interval
		}
	}
}
