package sim

import (
	"fmt"
	"math"
	"strings"
	"time"
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
	FixedUS              int64  `json:"fixed_us"`
	PerVisibleRequestUS  int64  `json:"per_visible_request_us,omitempty"`
	PerPendingTransferUS int64  `json:"per_pending_transfer_us,omitempty"`
	Provenance           string `json:"provenance"`
}

func (c LinearDecisionCost) Validate() error {
	if c.FixedUS < 0 || c.PerVisibleRequestUS < 0 || c.PerPendingTransferUS < 0 || strings.TrimSpace(c.Provenance) == "" {
		return fmt.Errorf("decision extra cost requires nonnegative coefficients and provenance")
	}
	return nil
}

func (c LinearDecisionCost) Estimate(v DecisionView, _ DecisionPlan) (DecisionCostEstimate, error) {
	if err := c.Validate(); err != nil {
		return DecisionCostEstimate{}, err
	}
	if c.PerPendingTransferUS > 0 && !v.KV.TransfersKnown {
		return DecisionCostEstimate{}, fmt.Errorf("decision transfer cost requires a known pending-transfer view")
	}
	extra := c.FixedUS
	for _, term := range []struct{ count, rate int64 }{
		{int64(len(v.Waiting)), c.PerVisibleRequestUS},
		{int64(len(v.Running)), c.PerVisibleRequestUS},
		{int64(len(v.KV.PendingTransfers)), c.PerPendingTransferUS},
	} {
		if term.count > 0 && term.rate > (math.MaxInt64-extra)/term.count {
			return DecisionCostEstimate{}, fmt.Errorf("decision extra cost overflows")
		}
		extra += term.count * term.rate
	}
	return DecisionCostEstimate{ExtraUS: extra, Provenance: c.Provenance}, nil
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
	if !valid {
		d.record.RuntimeWallNS += time.Since(start).Nanoseconds()
		s.deliverDecision(DecisionFeedback{Version: d.record.View.Version, Status: "superseded", Reason: "request_membership_changed"})
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
		// If the old plan could not admit anything and no I/O owns the engine,
		// the excluded arrivals still deserve their first formation attempt.
		s.ScheduleStepIfIdle(e.At)
	}
}
