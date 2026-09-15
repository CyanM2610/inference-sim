package sim

import (
	"fmt"
	"sort"
)

// DecodeFillPolicy retains a balanced prefill cohort while its real prefix
// loads are pending, issuing decode work in the meantime. Whole-block promotion
// and the configured token-ratio model are not Strata's layer-wise executor.
type DecodeFillPolicy struct {
	base                         *BalancedBatchPolicy
	prepared, proposed           *decodeFillBatch
	failed, nextFailed, attempts map[string]string
	version                      uint64
	mode                         string
}

type decodeFillBatch struct {
	waiting, resident []string
	endpoints         map[string]int64
}

func NewDecodeFillPolicy(base *BalancedBatchPolicy) (*DecodeFillPolicy, error) {
	if base == nil {
		return nil, fmt.Errorf("decode fill requires balanced batch selection")
	}
	return &DecodeFillPolicy{base: base, failed: map[string]string{}}, nil
}

func (p *DecodeFillPolicy) Reset() {
	p.prepared, p.proposed, p.attempts = nil, nil, nil
	p.failed = map[string]string{}
	p.nextFailed = nil
	p.version, p.mode = 0, ""
}

func (*DecodeFillPolicy) AdditionalCostCoverage() string {
	return "decode_fill_policy_and_control_not_independently_calibrated"
}

// DecodeFillState contains only committed policy state, never reservations.
type DecodeFillState struct {
	Waiting     []string          `json:"waiting"`
	Resident    []string          `json:"resident"`
	Endpoints   map[string]int64  `json:"endpoints"`
	RetryGuards map[string]string `json:"retry_guards"`
}

func (p *DecodeFillPolicy) State() DecodeFillState {
	s := DecodeFillState{Waiting: []string{}, Resident: []string{}, Endpoints: map[string]int64{}, RetryGuards: map[string]string{}}
	if p.prepared != nil {
		s.Waiting = append(s.Waiting, p.prepared.waiting...)
		s.Resident = append(s.Resident, p.prepared.resident...)
		for id, end := range p.prepared.endpoints {
			s.Endpoints[id] = end
		}
	}
	for id, signature := range p.failed {
		s.RetryGuards[id] = signature
	}
	return s
}

func decodeFillSignature(v DecisionView, s DecisionKVRequest) string {
	result := fmt.Sprintf("%d/%d/%d", v.HBMFreeBlocks, s.LocalPrefixBlocks, s.RecoverablePrefixBlocks)
	pools := append([]DecisionPool(nil), v.KV.Pools...)
	sort.Slice(pools, func(i, j int) bool { return pools[i].ID < pools[j].ID })
	for _, pool := range pools {
		result += fmt.Sprintf("|%s:%d:%d:%d", pool.ID, pool.FreeBlocks, pool.ReservedBlocks, pool.ProtectedBlocks)
	}
	return result
}

func (p *DecodeFillPolicy) Decide(v DecisionView) DecisionPlan {
	if !v.Capabilities.PrefillDeferrals || !v.Capabilities.Promotions || v.Capabilities.PromotionRetention != "request" || !v.KV.TransfersKnown || v.BlockTokens <= 0 {
		panic("decode fill requires round deferrals, request-retained promotion and current transfer state")
	}
	p.version, p.mode, p.proposed = v.Version, "ordinary", nil
	p.attempts, p.nextFailed = map[string]string{}, map[string]string{}
	live, states := map[string]DecisionRequest{}, map[string]DecisionKVRequest{}
	for _, group := range [][]DecisionRequest{v.Running, v.Waiting} {
		for _, r := range group {
			live[r.ID] = r
		}
	}
	for _, s := range v.KV.Requests {
		states[s.ID] = s
	}
	for id, signature := range p.failed {
		if _, ok := live[id]; ok {
			p.nextFailed[id] = signature
		}
	}
	decode, holdable := false, true
	var holds []string
	for _, r := range v.Running {
		s := states[r.ID]
		isDecode := r.EmittedTokens > 0 && r.ComputedTokens >= max(r.InputTokens, r.RecomputeUntilTokens) && !s.TransferPending && !s.Deferred
		decode = decode || isDecode
		if r.PrefillDeferrable {
			holds = append(holds, r.ID)
		} else if !isDecode && !s.TransferPending && !s.Deferred {
			holdable = false
		}
	}
	holdPlan := func() DecisionPlan {
		plan := DecisionPlan{Version: v.Version, Admission: &DecisionAdmission{Requests: []string{}}, PrefillDeferrals: holds}
		for _, r := range v.Running {
			s := states[r.ID]
			if r.EmittedTokens > 0 && r.ComputedTokens >= max(r.InputTokens, r.RecomputeUntilTokens) && !s.TransferPending && !s.Deferred {
				plan.TokenCaps = append(plan.TokenCaps, DecisionTokenCap{Request: r.ID, Tokens: 1})
			}
		}
		return plan
	}
	if p.prepared != nil {
		batch := &decodeFillBatch{endpoints: map[string]int64{}}
		for _, id := range p.prepared.waiting {
			if r, ok := live[id]; ok && r.EmittedTokens == 0 && r.ComputedTokens < r.InputTokens {
				batch.waiting = append(batch.waiting, id)
			}
		}
		for _, id := range p.prepared.resident {
			if r, ok := live[id]; ok && r.PrefillDeferrable {
				batch.resident = append(batch.resident, id)
			}
		}
		pending, ready := false, true
		for _, id := range batch.waiting {
			if end, ok := p.prepared.endpoints[id]; ok {
				batch.endpoints[id] = end
				s := states[id]
				pending = pending || s.TransferPending || s.Deferred
				ready = ready && s.LocalPrefixBlocks >= end && !s.TransferPending && !s.Deferred
			}
		}
		if len(batch.waiting)+len(batch.resident) > 0 {
			if pending && decode && holdable {
				p.proposed, p.mode = batch, "pending"
				return holdPlan()
			}
			if !pending && ready {
				// Restrict admission to the retained cohort, then restore a full
				// waiting permutation for the runtime's validation contract.
				view := cloneDecisionView(v)
				view.Waiting = nil
				selected := map[string]bool{}
				for _, id := range batch.waiting {
					selected[id] = true
				}
				for _, r := range v.Waiting {
					if selected[r.ID] {
						view.Waiting = append(view.Waiting, r)
					}
				}
				plan := p.base.Decide(view)
				for _, r := range v.Waiting {
					if !selected[r.ID] {
						plan.QueueOrder = append(plan.QueueOrder, r.ID)
					}
				}
				p.proposed, p.mode = batch, "resume"
				return plan
			}
		}
		// Partial failure, loss of decode, or phase changes return to the
		// ordinary runtime path without cancelling already submitted DMA.
		return p.base.Decide(v)
	}
	if !decode || !holdable {
		return p.base.Decide(v)
	}
	view := cloneDecisionView(v)
	// Promotion must leave the logits block to computation. Use that same
	// endpoint when selecting and costing the prospective prefill cohort.
	for i, s := range view.KV.Requests {
		if r, ok := live[s.ID]; ok {
			s.RecoverablePrefixBlocks = max(s.LocalPrefixBlocks, min(s.RecoverablePrefixBlocks, max(int64(0), r.InputTokens-1)/v.BlockTokens))
			view.KV.Requests[i] = s
		}
	}
	plan := p.base.Decide(view)
	batch := &decodeFillBatch{waiting: append([]string(nil), plan.Admission.Requests...), resident: append([]string(nil), holds...), endpoints: map[string]int64{}}
	var load, compute int64
	for _, cap := range plan.TokenCaps {
		compute += cap.Tokens
	}
	keys := map[string]bool{}
	for _, restore := range plan.Restores {
		r, s := live[restore.Request], states[restore.Request]
		end := restore.MaxPrefixBlocks
		if end <= s.LocalPrefixBlocks {
			continue
		}
		if r.ComputedTokens != 0 || r.EmittedTokens != 0 || s.TransferPending || s.Deferred {
			return p.base.Decide(v)
		}
		signature := decodeFillSignature(v, s)
		if p.failed[r.ID] == signature {
			return p.base.Decide(v)
		}
		batch.endpoints[r.ID] = end
		p.attempts[r.ID] = signature
		if p.base.bundleHits {
			s.RecoverablePrefixBlocks = end
			ids, usable := decisionRestoreKeys(s)
			if !usable {
				return p.base.Decide(v)
			}
			for _, id := range ids {
				if !keys[id] {
					load += v.BlockTokens
					keys[id] = true
				}
			}
		} else {
			load += (end - s.LocalPrefixBlocks) * v.BlockTokens
		}
	}
	if load == 0 || float64(load) <= p.base.MaxLoadComputeRatio*float64(compute) {
		p.attempts = nil
		return p.base.Decide(v)
	}
	result := holdPlan()
	for _, id := range batch.waiting {
		if end, ok := batch.endpoints[id]; ok {
			result.Promotions = append(result.Promotions, DecisionPromotion{Request: id, MaxPrefixBlocks: end})
		}
	}
	p.proposed, p.mode = batch, "submit"
	return result
}

func (p *DecodeFillPolicy) Observe(f DecisionFeedback) {
	if f.Version != p.version {
		panic("decode fill feedback version mismatch")
	}
	if f.Status != "applied" {
		return
	}
	p.failed = p.nextFailed
	if p.mode == "submit" {
		outcomes := map[string]DecisionPromotionOutcome{}
		for _, o := range f.Promotions {
			outcomes[o.Request] = o
		}
		accepted := false
		for id, signature := range p.attempts {
			o, ok := outcomes[id]
			if !ok {
				panic("decode fill lacks actual promotion feedback")
			}
			accepted = accepted || o.ReadyBlocks+o.PendingBlocks+o.StartedBlocks > 0
			if o.BlockedBlocks > 0 {
				p.failed[id] = signature
			}
		}
		if !accepted {
			p.proposed = nil
		}
	}
	if p.mode == "resume" && p.proposed != nil {
		granted := map[string]bool{}
		for _, g := range f.Grants {
			if g.Tokens > 0 {
				granted[g.Request] = true
			}
		}
		progress := false
		remaining := func(ids []string) []string {
			var result []string
			for _, id := range ids {
				if granted[id] {
					progress = true
					delete(p.proposed.endpoints, id)
				} else {
					result = append(result, id)
				}
			}
			return result
		}
		p.proposed.waiting = remaining(p.proposed.waiting)
		p.proposed.resident = remaining(p.proposed.resident)
		if !progress || len(p.proposed.waiting)+len(p.proposed.resident) == 0 {
			p.proposed = nil
		}
	}
	p.prepared = p.proposed
}
