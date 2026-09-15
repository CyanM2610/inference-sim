package sim

import (
	"fmt"
	"math"
	"sort"
)

// QueueDecisionPolicy is a small policy-only caller of the joint plan boundary.
// It never allocates blocks, modifies request progress or records TTFT.
type QueueDecisionPolicy struct {
	RestorePrefixBlocks *int64
	Order               string
	PrefillTokenCap     int64
	AdmissionDelayUS    int64
}

func NewQueueDecisionPolicy(order string, cap int64) (*QueueDecisionPolicy, error) {
	if order == "" {
		order = "fcfs"
	}
	if order != "fcfs" && order != "sjf" && order != "edf" {
		return nil, fmt.Errorf("unknown decision queue order %q", order)
	}
	if cap < 0 {
		return nil, fmt.Errorf("negative decision prefill cap")
	}
	return &QueueDecisionPolicy{Order: order, PrefillTokenCap: cap}, nil
}
func (p *QueueDecisionPolicy) Decide(v DecisionView) DecisionPlan {
	requests := append([]DecisionRequest(nil), v.Waiting...)
	deadline := func(r DecisionRequest) int64 {
		if r.TTFTTargetUS <= 0 {
			return math.MaxInt64
		}
		if r.ArrivalUS > math.MaxInt64-r.TTFTTargetUS {
			return math.MaxInt64
		}
		return r.ArrivalUS + r.TTFTTargetUS
	}
	if p.Order != "fcfs" {
		sort.SliceStable(requests, func(i, j int) bool {
			a, b := requests[i], requests[j]
			if p.Order == "sjf" && a.InputTokens != b.InputTokens {
				return a.InputTokens < b.InputTokens
			}
			if p.Order == "edf" && deadline(a) != deadline(b) {
				return deadline(a) < deadline(b)
			}
			if a.ArrivalUS != b.ArrivalUS {
				return a.ArrivalUS < b.ArrivalUS
			}
			return a.ID < b.ID
		})
	}
	plan := DecisionPlan{Version: v.Version}
	pending := map[string]bool{}
	for _, r := range v.KV.Requests {
		pending[r.ID] = r.TransferPending
	}
	for _, r := range requests {
		plan.QueueOrder = append(plan.QueueOrder, r.ID)
		if p.RestorePrefixBlocks != nil && !pending[r.ID] {
			if v.BlockTokens <= 0 {
				panic("restore decision requires a positive block size")
			}
			plan.Restores = append(plan.Restores, DecisionRestore{Request: r.ID, MaxPrefixBlocks: min(*p.RestorePrefixBlocks, r.InputTokens/v.BlockTokens)})
		}
		if p.AdmissionDelayUS > 0 {
			if r.ArrivalUS > math.MaxInt64-p.AdmissionDelayUS {
				panic("admission delay deadline overflows")
			}
			until := r.ArrivalUS + p.AdmissionDelayUS
			if until > v.NowUS && until != r.WaitUntilUS {
				plan.Waits = append(plan.Waits, DecisionWait{Request: r.ID, UntilUS: until})
			}
		}
	}
	if p.PrefillTokenCap > 0 {
		for _, r := range append(requests, v.Running...) {
			if r.ComputedTokens < r.InputTokens {
				plan.TokenCaps = append(plan.TokenCaps, DecisionTokenCap{Request: r.ID, Tokens: p.PrefillTokenCap})
			}
		}
	}
	return plan
}
func (*QueueDecisionPolicy) Observe(DecisionFeedback) {}
