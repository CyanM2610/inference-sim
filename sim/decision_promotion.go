package sim

import "fmt"

// DecisionPromotion warms a waiting request's existing prefix independently of
// compute admission. The endpoint excludes the final logits token. It does not
// choose a source tier or cancel I/O. Residence after adoption follows the
// backend's declared PromotionRetention capability.
type DecisionPromotion struct {
	Request         string `json:"request"`
	MaxPrefixBlocks int64  `json:"max_prefix_blocks"`
}

// Counts partition the requested prefix at application time. Started blocks
// have real reservations/I/O, not ready data. Blocked suffixes are not retried
// implicitly. Completion is observable through normal transfer/cache events.
type DecisionPromotionOutcome struct {
	Request         string `json:"request"`
	MaxPrefixBlocks int64  `json:"max_prefix_blocks"`
	ReadyBlocks     int64  `json:"ready_blocks"`
	PendingBlocks   int64  `json:"pending_blocks"`
	StartedBlocks   int64  `json:"started_blocks"`
	BlockedBlocks   int64  `json:"blocked_blocks"`
	Reason          string `json:"reason"`
}

// PromotionDecisionStore receives request tokens only from the trusted runtime;
// a policy supplies an ID and endpoint, never physical blocks or future input.
type PromotionDecisionStore interface {
	PromotionDecisionsEnabled() bool
	PromotionRetention() string
	PromoteRequestPrefix(DecisionKVQuery, int64) DecisionPromotionOutcome
}

func validateDecisionPromotions(v DecisionView, p DecisionPlan) error {
	if len(p.Promotions) == 0 {
		return nil
	}
	if !v.Capabilities.Promotions || v.BlockTokens <= 0 {
		return fmt.Errorf("request prefix promotion is unsupported")
	}
	eligible := map[string]int64{}
	for _, r := range v.Waiting {
		if r.ComputedTokens == 0 && r.EmittedTokens == 0 {
			eligible[r.ID] = max(0, r.InputTokens-1) / v.BlockTokens
		}
	}
	seen := map[string]bool{}
	for _, action := range p.Promotions {
		maxBlocks, ok := eligible[action.Request]
		if !ok || seen[action.Request] || action.MaxPrefixBlocks <= 0 || action.MaxPrefixBlocks > maxBlocks {
			return fmt.Errorf("invalid prefix promotion for %q", action.Request)
		}
		seen[action.Request] = true
	}
	for _, restore := range p.Restores {
		if seen[restore.Request] {
			return fmt.Errorf("promotion and restore limits cannot target the same request in one plan")
		}
	}
	return nil
}

func (s *Simulator) applyDecisionPromotions(plan DecisionPlan) {
	if len(plan.Promotions) == 0 {
		return
	}
	backend, ok := s.KVCache.(PromotionDecisionStore)
	if !ok || !backend.PromotionDecisionsEnabled() {
		panic("request prefix promotion backend disappeared")
	}
	requests := map[string]*Request{}
	for _, r := range s.WaitQ.Items() {
		requests[r.ID] = r
	}
	for _, action := range plan.Promotions {
		r := requests[action.Request]
		if r == nil {
			panic("promotion request missing after membership validation")
		}
		outcome := backend.PromoteRequestPrefix(DecisionKVQuery{ID: r.ID, Input: append([]TokenID(nil), r.FullInputTokens()...)}, action.MaxPrefixBlocks)
		s.decision.record.Feedback.Promotions = append(s.decision.record.Feedback.Promotions, outcome)
	}
}
