package sim

import (
	"fmt"
	"sort"
)

// DecisionPrefixProducer describes input identity, not a ready copy, reservation
// or promise of completion. Only arrived waiting/running requests occur here.
// CommonPrefixBlocks excludes the consumer's final query token. Progress and
// actual ready/recoverable prefixes must be read separately from DecisionView.
type DecisionPrefixProducer struct {
	Consumer           string `json:"consumer"`
	Producer           string `json:"producer"`
	CommonPrefixBlocks int64  `json:"common_prefix_blocks"`
}

// EnableDecisionPrefixProducers opts into potentially quadratic snapshot work.
// New matching work is measured as snapshot wall time, not silently calibrated.
func (s *Simulator) EnableDecisionPrefixProducers() error {
	if s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents {
		return fmt.Errorf("prefix producers require a pre-run decision policy and batch completion events")
	}
	if _, ok := s.KVCache.(DecisionStateProvider); !ok {
		return fmt.Errorf("prefix producers require a ready-prefix state provider")
	}
	s.decision.prefixProducers = true
	return nil
}

func decisionPrefixProducers(waiting []DecisionRequest, queries []DecisionKVQuery, block int64) []DecisionPrefixProducer {
	inputs := make(map[string][]TokenID, len(queries))
	for _, q := range queries {
		inputs[q.ID] = q.Input
	}
	var out []DecisionPrefixProducer
	for _, c := range waiting {
		input := inputs[c.ID]
		for _, p := range queries {
			if c.ID == p.ID {
				continue
			}
			end := min(len(input)-1, len(p.Input))
			i := 0
			for i < end && input[i] == p.Input[i] {
				i++
			}
			if n := int64(i) / block; n > 0 {
				out = append(out, DecisionPrefixProducer{c.ID, p.ID, n})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Consumer != out[j].Consumer {
			return out[i].Consumer < out[j].Consumer
		}
		return out[i].Producer < out[j].Producer
	})
	return out
}

// ProducerWaitPolicy defers untouched consumers while an arrived producer can
// provide a longer shared prefix. The arrival-relative age limit is checked at
// each ordinary scheduler step; it is not an exact timer or hard SLO guarantee.
// Running work is never held. Waiting dependencies point strictly backward in
// arrival/ID order, so even a chain has an unheld root. No future KV is pinned.
type ProducerWaitPolicy struct {
	base                       DecisionPolicy
	minSharedTokens, maxWaitUS int64
}

func NewProducerWaitPolicy(base DecisionPolicy, minSharedTokens, maxWaitUS int64) (*ProducerWaitPolicy, error) {
	if base == nil || minSharedTokens <= 0 || maxWaitUS <= 0 {
		return nil, fmt.Errorf("producer waiting requires a base policy and positive shared-token/age limits")
	}
	return &ProducerWaitPolicy{base, minSharedTokens, maxWaitUS}, nil
}

func (p *ProducerWaitPolicy) Decide(v DecisionView) DecisionPlan {
	if !v.Capabilities.PrefixProducers || !v.Capabilities.AdmissionSelection || !v.KV.PrefixStateKnown || v.BlockTokens <= 0 {
		panic("producer waiting requires explicit prefix producers, admission and ready-prefix state")
	}
	requests := map[string]DecisionRequest{}
	running := map[string]bool{}
	states := map[string]DecisionKVRequest{}
	for _, r := range v.Waiting {
		requests[r.ID] = r
	}
	for _, r := range v.Running {
		requests[r.ID] = r
		running[r.ID] = true
	}
	for _, r := range v.KV.Requests {
		states[r.ID] = r
	}
	held := map[string]bool{}
	for _, edge := range v.PrefixProducers {
		c, cok := requests[edge.Consumer]
		producer, pok := requests[edge.Producer]
		state, known := states[c.ID]
		if !cok || !pok || !known || running[c.ID] || c.ID == producer.ID || c.ComputedTokens > 0 || state.TransferPending || state.Deferred {
			continue
		}
		if c.WaitUntilUS > v.NowUS || v.NowUS-c.ArrivalUS >= p.maxWaitUS {
			continue
		}
		if edge.CommonPrefixBlocks < 1+(p.minSharedTokens-1)/v.BlockTokens || max(state.LocalPrefixBlocks, state.RecoverablePrefixBlocks) >= edge.CommonPrefixBlocks {
			continue
		}
		if !running[producer.ID] && (producer.WaitUntilUS > v.NowUS || producer.ArrivalUS > c.ArrivalUS || producer.ArrivalUS == c.ArrivalUS && producer.ID >= c.ID) {
			continue
		}
		held[c.ID] = true
	}
	filtered := v
	filtered.Waiting = nil
	for _, r := range v.Waiting {
		if !held[r.ID] {
			filtered.Waiting = append(filtered.Waiting, r)
		}
	}
	plan := p.base.Decide(filtered)
	// Expand the base's permutation to the complete runtime queue. Restrict only
	// this step's admission; no zero quotas, fabricated time or resource mutation.
	if len(held) > 0 {
		if plan.QueueOrder == nil {
			for _, r := range filtered.Waiting {
				plan.QueueOrder = append(plan.QueueOrder, r.ID)
			}
		}
		if plan.Admission == nil {
			plan.Admission = &DecisionAdmission{Requests: []string{}}
			for _, r := range filtered.Waiting {
				plan.Admission.Requests = append(plan.Admission.Requests, r.ID)
			}
		}
		for _, r := range v.Waiting {
			if held[r.ID] {
				plan.QueueOrder = append(plan.QueueOrder, r.ID)
			}
		}
	}
	return plan
}

func (p *ProducerWaitPolicy) Observe(f DecisionFeedback) { p.base.Observe(f) }
