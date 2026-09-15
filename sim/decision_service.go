package sim

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

const ExplicitDecisionCostCoverage = "explicit_extra_service_sim_time_uncalibrated"

// DecisionCostModel estimates additional CPU service beyond the base engine
// profile. Only detached, currently visible inputs are supplied. Host wall time
// is never implicitly converted to simulation time.
type DecisionCostModel interface {
	Estimate(DecisionView, DecisionPlan) (DecisionCostEstimate, error)
}

type DecisionCostEstimate struct {
	ExtraUS    int64
	Provenance string
}

func (c DecisionCostEstimate) validate(now int64) error {
	if c.ExtraUS < 0 || strings.TrimSpace(c.Provenance) == "" {
		return fmt.Errorf("decision cost requires nonnegative extra_us and provenance")
	}
	if now > math.MaxInt64-c.ExtraUS {
		return fmt.Errorf("decision service completion time overflows")
	}
	return nil
}

type DecisionService struct {
	StartUS    int64  `json:"start_us"`
	EndUS      int64  `json:"end_us"`
	ExtraUS    int64  `json:"extra_us"`
	Provenance string `json:"provenance"`
}

// LinearDecisionCost is an explicit sensitivity model, not a calibration claim.
// FixedUS applies once per decision, including decisions later superseded.
type LinearDecisionCost struct {
	profileWarning       func(parameter string, observed, maximum int64)
	PerPrefillWaitUS     int64               `json:"per_prefill_wait_us,omitempty"`
	PerRestoreChoiceUS   int64               `json:"per_restore_choice_us,omitempty"`
	PerPreemptionUS      int64               `json:"per_preemption_us,omitempty"`
	PerPrefixBlockUS     int64               `json:"per_prefix_block_us,omitempty"`
	Limits               *DecisionCostLimits `json:"limits,omitempty"`
	FixedUS              int64               `json:"fixed_us"`
	PerVisibleRequestUS  int64               `json:"per_visible_request_us,omitempty"`
	PerPendingTransferUS int64               `json:"per_pending_transfer_us,omitempty"`
	PerPromotionUS       int64               `json:"per_promotion_us,omitempty"`
	PerPromotionBlockUS  int64               `json:"per_promotion_block_us,omitempty"`
	Provenance           string              `json:"provenance"`
}

// DecisionCostLimits bounds a declared partial profile's observed state shapes.
// Numeric bounds produce extrapolation warnings; capability and data contracts
// remain errors. Pricing uses actual work with the original coefficients.
type DecisionCostLimits struct {
	RequirePrefillWaits       bool   `json:"require_prefill_waits,omitempty"`
	MaxRestoreChoices         int64  `json:"max_restore_choices,omitempty"`
	RequirePrefillPreemption  bool   `json:"require_prefill_preemption,omitempty"`
	MaxPreemptions            int64  `json:"max_preemptions,omitempty"`
	MaxBatchTokenCap          int64  `json:"max_batch_token_cap,omitempty"`
	RequirePrefixSharing      bool   `json:"require_prefix_sharing,omitempty"`
	MaxPrefixBlocks           int64  `json:"max_prefix_blocks,omitempty"`
	MaxPromotionActions       int64  `json:"max_promotion_actions,omitempty"`
	MaxPromotionBlocks        int64  `json:"max_promotion_blocks,omitempty"`
	MaxOutputTokens           int64  `json:"max_output_tokens,omitempty"`
	MaxVisibleRequests        int64  `json:"max_visible_requests"`
	MaxPendingTransfers       int64  `json:"max_pending_transfers"`
	MaxInputTokens            int64  `json:"max_input_tokens"`
	RequirePrefixProducers    bool   `json:"require_prefix_producers,omitempty"`
	RequireBatchPrefixReuse   bool   `json:"require_batch_prefix_reuse,omitempty"`
	RequirePromotionRetention string `json:"require_promotion_retention,omitempty"`
}

func (l DecisionCostLimits) validate() error {
	if l.MaxRestoreChoices < 0 {
		return fmt.Errorf("negative restore choice coverage")
	}
	if l.MaxPreemptions < 0 || l.MaxPreemptions > 0 && !l.RequirePrefillPreemption {
		return fmt.Errorf("preemption cost limit requires declared prefill preemption coverage")
	}
	if l.MaxBatchTokenCap < 0 {
		return fmt.Errorf("batch token cap cost bound cannot be negative")
	}
	if l.MaxPrefixBlocks < 0 || l.RequirePrefixSharing && l.MaxPrefixBlocks == 0 {
		return fmt.Errorf("prefix sharing cost coverage requires a positive block limit")
	}
	if l.MaxPromotionActions < 0 || l.MaxPromotionBlocks < 0 {
		return fmt.Errorf("decision cost promotion limits cannot be negative")
	}
	if l.RequirePromotionRetention != "" && l.RequirePromotionRetention != "cache" && l.RequirePromotionRetention != "request" {
		return fmt.Errorf("decision cost promotion retention must be cache or request")
	}
	if l.MaxVisibleRequests <= 0 || l.MaxPendingTransfers < 0 || l.MaxInputTokens <= 0 || l.MaxOutputTokens < 0 {
		return fmt.Errorf("decision cost limits require positive request/input limits and nonnegative pending transfers")
	}
	return nil
}

func (l DecisionCostLimits) check(v DecisionView, warn func(string, int64, int64)) error {
	if v.Capabilities.PrefillWaits != l.RequirePrefillWaits {
		return fmt.Errorf("decision cost profile does not cover resident prefill waits")
	}
	if l.MaxRestoreChoices > 0 {
		if v.Estimates == nil {
			return fmt.Errorf("restore choice cost requires estimates")
		}
		var count int64
		for _, r := range v.Estimates.Requests {
			count += int64(len(r.Choices))
		}
		warn("restore_choices", count, l.MaxRestoreChoices)
	}
	if v.Capabilities.PrefillPreemption != l.RequirePrefillPreemption {
		return fmt.Errorf("decision cost profile does not cover prefill preemption observation")
	}
	if v.KV.PrefixSharingKnown != l.RequirePrefixSharing {
		return fmt.Errorf("decision cost profile does not match prefix sharing observation work")
	}
	if l.RequirePrefixSharing {
		var blocks int64
		for _, r := range v.KV.Requests {
			if int64(len(r.PrefixBlocks)) > math.MaxInt64-blocks {
				return fmt.Errorf("decision prefix block count overflow")
			}
			blocks += int64(len(r.PrefixBlocks))
		}
		warn("prefix_blocks", blocks, l.MaxPrefixBlocks)
	}
	if l.RequirePromotionRetention != "" && (!v.Capabilities.Promotions || v.Capabilities.PromotionRetention != l.RequirePromotionRetention) {
		return fmt.Errorf("decision cost profile requires matching promotion retention")
	}
	if !v.KV.TransfersKnown {
		return fmt.Errorf("decision cost requires known transfer state")
	}
	warn("visible_requests", int64(len(v.Waiting)+len(v.Running)), l.MaxVisibleRequests)
	warn("pending_transfers", int64(len(v.KV.PendingTransfers)), l.MaxPendingTransfers)
	for _, group := range [][]DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			if r.InputTokens < 0 {
				return fmt.Errorf("negative decision input tokens")
			}
			warn("input_tokens", r.InputTokens, l.MaxInputTokens)
			if l.MaxOutputTokens > 0 {
				if r.ClientOutputLimit <= 0 {
					return fmt.Errorf("decision output budget is unknown")
				}
				warn("client_output_limit", int64(r.ClientOutputLimit), l.MaxOutputTokens)
			}
		}
	}
	if l.RequirePrefixProducers && !v.Capabilities.PrefixProducers || l.RequireBatchPrefixReuse && (!v.Capabilities.BatchPrefixReuseKnown || !v.Capabilities.BatchPrefixReuse) {
		return fmt.Errorf("decision cost profile requires declared prefix producer/batch reuse capabilities")
	}
	return nil
}

func (c LinearDecisionCost) Validate() error {
	if c.PerPrefillWaitUS < 0 {
		return fmt.Errorf("negative prefill wait coefficient")
	}
	if c.PerRestoreChoiceUS < 0 {
		return fmt.Errorf("negative restore choice coefficient")
	}
	if c.Limits != nil {
		if err := c.Limits.validate(); err != nil {
			return err
		}
	}
	if c.FixedUS < 0 || c.PerVisibleRequestUS < 0 || c.PerPendingTransferUS < 0 || c.PerPromotionUS < 0 || c.PerPromotionBlockUS < 0 || c.PerPrefixBlockUS < 0 || c.PerPreemptionUS < 0 || strings.TrimSpace(c.Provenance) == "" {
		return fmt.Errorf("decision extra cost requires nonnegative coefficients and provenance")
	}
	return nil
}

func (c LinearDecisionCost) Estimate(v DecisionView, plan DecisionPlan) (DecisionCostEstimate, error) {
	if err := c.Validate(); err != nil {
		return DecisionCostEstimate{}, err
	}
	if c.Limits != nil {
		if err := c.Limits.check(v, c.warnProfile); err != nil {
			return DecisionCostEstimate{}, err
		}
		c.warnProfile("batch_token_cap", plan.BatchTokenCap, c.Limits.MaxBatchTokenCap)
		c.warnProfile("preemptions", int64(len(plan.Preemptions)), c.Limits.MaxPreemptions)
		if limit := c.Limits.MaxPromotionActions; limit > 0 {
			c.warnProfile("promotion_actions", int64(len(plan.Promotions)), limit)
		}
		if limit := c.Limits.MaxPromotionBlocks; limit > 0 {
			var blocks int64
			for _, action := range plan.Promotions {
				if action.MaxPrefixBlocks < 0 || action.MaxPrefixBlocks > math.MaxInt64-blocks {
					return DecisionCostEstimate{}, fmt.Errorf("invalid or overflowing promotion blocks")
				}
				blocks += action.MaxPrefixBlocks
			}
			c.warnProfile("promotion_blocks", blocks, limit)
		}
	}
	if c.PerPendingTransferUS > 0 && !v.KV.TransfersKnown {
		return DecisionCostEstimate{}, fmt.Errorf("decision transfer cost requires a known pending-transfer view")
	}
	if c.PerPrefixBlockUS > 0 && !v.KV.PrefixSharingKnown {
		return DecisionCostEstimate{}, fmt.Errorf("prefix block cost requires known sharing work")
	}
	if c.PerPromotionUS > 0 || c.PerPromotionBlockUS > 0 {
		if err := validateDecisionPromotions(v, plan); err != nil {
			return DecisionCostEstimate{}, err
		}
	}
	if err := validateDecisionPreemptions(v, plan); err != nil {
		return DecisionCostEstimate{}, err
	}
	extra := c.FixedUS
	if c.PerRestoreChoiceUS > 0 {
		if v.Estimates == nil {
			return DecisionCostEstimate{}, fmt.Errorf("restore choice cost requires estimates")
		}
		for _, r := range v.Estimates.Requests {
			n := int64(len(r.Choices))
			if n > 0 && c.PerRestoreChoiceUS > (math.MaxInt64-extra)/n {
				return DecisionCostEstimate{}, fmt.Errorf("restore choice cost overflow")
			}
			extra += n * c.PerRestoreChoiceUS
		}
	}
	for _, term := range []struct{ count, rate int64 }{
		{prefillWaitUpdates(v, plan), c.PerPrefillWaitUS},
		{int64(len(v.Waiting)), c.PerVisibleRequestUS},
		{int64(len(v.Running)), c.PerVisibleRequestUS},
		{int64(len(v.KV.PendingTransfers)), c.PerPendingTransferUS},
		{int64(len(plan.Promotions)), c.PerPromotionUS},
		{int64(len(plan.Preemptions)), c.PerPreemptionUS},
	} {
		if term.count > 0 && term.rate > (math.MaxInt64-extra)/term.count {
			return DecisionCostEstimate{}, fmt.Errorf("decision extra cost overflows")
		}
		extra += term.count * term.rate
	}
	for _, action := range plan.Promotions {
		if c.PerPromotionBlockUS > 0 && action.MaxPrefixBlocks > (math.MaxInt64-extra)/c.PerPromotionBlockUS {
			return DecisionCostEstimate{}, fmt.Errorf("decision promotion block cost overflows")
		}
		extra += action.MaxPrefixBlocks * c.PerPromotionBlockUS
	}
	for _, r := range v.KV.Requests {
		count := int64(len(r.PrefixBlocks))
		if count > 0 && c.PerPrefixBlockUS > (math.MaxInt64-extra)/count {
			return DecisionCostEstimate{}, fmt.Errorf("decision prefix block cost overflows")
		}
		extra += count * c.PerPrefixBlockUS
	}
	return DecisionCostEstimate{ExtraUS: extra, Provenance: c.Provenance}, nil
}

// SetProfileWarningObserver attaches run-local reporting without exposing the
// mutable model or input snapshot to the observer. It does not change pricing.
func (c *LinearDecisionCost) SetProfileWarningObserver(observer func(string, int64, int64)) {
	c.profileWarning = observer
}

func (c LinearDecisionCost) warnProfile(parameter string, observed, maximum int64) {
	if observed <= maximum {
		return
	}
	if c.profileWarning != nil {
		c.profileWarning(parameter, observed, maximum)
	} else {
		logrus.Warnf("profile extrapolation: linear_decision.%s observed=%d measured_max=%d; unchanged cost formula", parameter, observed, maximum)
	}
}

func (s *Simulator) SetDecisionCostModel(model DecisionCostModel) error {
	if model == nil || s.decision == nil || s.stepCount != 0 {
		return fmt.Errorf("decision cost model requires a policy installed before execution")
	}
	if !s.batchCompletionEvents {
		return fmt.Errorf("decision service requires batch completion events")
	}
	s.decision.cost = model
	return nil
}

// DecisionCompleteEvent ends CPU service; no resources were allocated by its
// plan before this event. Existing timeout, arrival and DMA events still run.
// Same-tick ordering matches StepEvent: arrivals precede it, timeouts follow it.
type DecisionCompleteEvent struct {
	At               int64
	waiting, running []*Request // trusted runtime membership, never policy input
}

func (e *DecisionCompleteEvent) Timestamp() int64 { return e.At }
func (e *DecisionCompleteEvent) Priority() int    { return PriorityStep }
func (e *DecisionCompleteEvent) Execute(s *Simulator) {
	d := s.decision
	if d == nil || d.pending != e {
		panic("unexpected decision service completion")
	}
	start := time.Now()
	d.pending = nil
	d.applying = true
	s.KVCache.SetClock(e.At)
	if d.record.ControlStep {
		s.applyDecisionPlan()
		s.finishDecisionControl(e.At)
		return
	}
	// Keep stepEvent non-nil through formation: KV reservation may issue wakes.
	waiting := make(map[*Request]bool, s.WaitQ.Len())
	for _, r := range s.WaitQ.Items() {
		waiting[r] = true
	}
	running := map[*Request]bool{}
	if s.RunningBatch != nil {
		for _, r := range s.RunningBatch.Requests {
			running[r] = true
		}
	}
	valid := len(running) == len(e.running)
	for _, r := range e.waiting {
		valid = valid && waiting[r]
	}
	for _, r := range e.running {
		valid = valid && running[r]
	}
	reason := "request_membership_changed"
	// A per-round hold may rely on a declared revisit or pending I/O which
	// disappears during paid CPU service. Replan without refunding that service.
	if valid && len(d.record.Plan.PrefillDeferrals) > 0 && validateDecisionPrefillDeferrals(s.decisionView(e.At), d.record.Plan) != nil {
		valid = false
		reason = "prefill_deferral_state_changed"
	}
	// A resident wait can expire while this service is being paid. A batch cap
	// which relied on that pause may now starve another resident (even decode).
	// Keep the paid service and obtain a fresh plan before any resource mutation.
	if valid && d.prefillWaits && d.record.Plan.BatchTokenCap > 0 && d.record.Plan.BatchTokenCap < minimumRunningGrants(s.decisionView(e.At), d.record.Plan) {
		valid = false
		reason = "prefill_wait_expired_batch_cap"
	}
	if !valid {
		d.record.RuntimeWallNS += time.Since(start).Nanoseconds()
		s.deliverDecision(DecisionFeedback{Version: d.record.View.Version, Status: "superseded", Reason: reason})
		d.applying = false
		s.flushDecisionEvents(e.At)
		if s.RunningBatch != nil && len(s.RunningBatch.Requests) == 0 {
			s.RunningBatch = nil
		}
		s.Step(e.At)
		return
	}
	eligible := make(map[*Request]bool, len(e.waiting))
	for _, r := range e.waiting {
		eligible[r] = true
	}
	late := s.WaitQ.Extract(func(r *Request) bool { return !eligible[r] })
	d.record.RuntimeWallNS += time.Since(start).Nanoseconds()
	caps := s.applyDecisionPlan()
	preemptionsBefore := s.Metrics.PreemptionCount
	s.formScheduledBatch(e.At, caps, late)
	d.applying = false
	s.flushDecisionEvents(e.At)
	s.executeScheduledBatch(e.At, preemptionsBefore)
	if len(late) > 0 {
		if d.executionPending != nil {
			d.executionPending.revisit = true
		}
		// If the old plan could not admit anything and no I/O owns the engine,
		// the excluded arrivals still deserve their first formation attempt.
		s.ScheduleStepIfIdle(e.At)
	}
}
