package policylab

import (
	"fmt"
	"math"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

// A nil coefficient vector declares an unmeasured action, not a free action.
// MaxCounts records the declared evaluated scope; it does not imply that every
// combination within that box was measured or has a certified error bound.
type QueueServiceCurve struct {
	CoefficientsUS []float64 `json:"coefficients_us"`
	MaxCounts      []int64   `json:"max_counts"`
}

type QueueServiceCostConfig struct {
	Spill *SpillServiceCostConfig `json:"spill,omitempty"`
	// The base stages exclude refresh work. This declared per-decision
	// increment is also charged on scans which produce no budget update.
	ReestimateExtraUS    int64                        `json:"reestimate_extra_us,omitempty"`
	ReestimateProvenance string                       `json:"reestimate_provenance,omitempty"`
	Family               string                       `json:"family"`
	EstimatorFamily      string                       `json:"estimator_family"`
	Stages               map[string]QueueServiceCurve `json:"stages"`
	Shape                RestoreServiceShape          `json:"shape"`
	HBMBlocks            int64                        `json:"hbm_blocks"`
	MaxInputTokens       int64                        `json:"max_input_tokens"`
	MaxOutputTokens      int                          `json:"max_output_tokens"`
	Provenance           string                       `json:"provenance"`
	Coverage             string                       `json:"coverage"`
}

type queueServiceCost struct{ profile QueueServiceCostConfig }

var queueStageSizes = map[string]int{"decision": 6, "reservation_view": 3, "estimate": 3,
	"execution": 7, "feedback": 5, "completion": 3, "registration": 1, "action": 1}

func newQueueServiceCost(c Config) (*queueServiceCost, error) {
	d := c.DecisionPolicy
	if d == nil || d.QueueServiceCost == nil || d.BudgetRestore == nil || c.EnginePhases == nil || c.DecisionEstimates == nil ||
		!c.RestoreControl || !c.DecodeCapacityReservation || !d.CapacityReservationView || !d.CapacityPreemption || !d.PrefillPreemption || !d.ControlSteps {
		return nil, fmt.Errorf("queue service requires the capacity/reservation budget path and explicit restore estimates")
	}
	if d.ExtraCost != nil || d.RestoreServiceCost != nil || d.WaitDecisionCost != nil || d.ExecutionCost != nil || d.WaitExecutionCost != nil ||
		d.HostServiceCost != nil || d.WaitHostServiceCost != nil || d.WaitPostStepCost != nil {
		return nil, fmt.Errorf("queue service cannot be added to another controller/host service profile")
	}
	if d.PrefillWaits || d.PrefillWaitProbe != nil || d.BalancedBatch != nil || d.ProducerWait != nil || d.ProducerPrefixes || d.PrefixSharing || c.BatchPrefixReuse || c.PromotionControl || c.Native != nil {
		return nil, fmt.Errorf("queue service profile does not cover waits, promotion, sharing or a different execution backend")
	}
	p := *d.QueueServiceCost
	if d.BudgetRestore.ReestimateOnDecodeDrop {
		if p.ReestimateExtraUS <= 0 || strings.TrimSpace(p.ReestimateProvenance) == "" {
			return nil, fmt.Errorf("queue service profile requires explicit additional budget reestimate service and provenance")
		}
	} else if p.ReestimateExtraUS != 0 || p.ReestimateProvenance != "" {
		return nil, fmt.Errorf("budget reestimate service requires the enabled policy option")
	}
	if (p.Family != "budget_pressure" && p.Family != "budget_queues") || p.Family != d.BudgetRestore.Mode ||
		len(c.Instances) != 1 || len(c.Pools) != 1 || p.HBMBlocks != c.Instances[0].HBMBlocks || p.Shape != serviceShape(c) ||
		p.MaxInputTokens <= 0 || p.MaxOutputTokens <= 0 || strings.TrimSpace(p.Provenance) == "" || strings.TrimSpace(p.Coverage) == "" {
		return nil, fmt.Errorf("queue service family, geometry, bounds or provenance mismatch")
	}
	wantEstimator := "plain_four"
	if c.EnginePhases.WorkerMetadata != nil {
		wantEstimator = "worker_six"
	}
	if p.EstimatorFamily != wantEstimator || c.BatchCost == nil || c.BatchCost.EnqueueUS != 0 {
		return nil, fmt.Errorf("queue service requires its measured estimator family and full registration replacing enqueue cost")
	}
	for _, r := range c.Requests {
		if int64(len(r.Input)) > p.MaxInputTokens || r.MaxOutputTokens <= 0 || r.MaxOutputTokens > p.MaxOutputTokens {
			return nil, fmt.Errorf("queue request exceeds declared input/output coverage")
		}
	}
	if len(p.Stages) != len(queueStageSizes) {
		return nil, fmt.Errorf("queue service requires all eight disjoint stages")
	}
	stages := make(map[string]QueueServiceCurve, len(p.Stages))
	for name, size := range queueStageSizes {
		x, ok := p.Stages[name]
		if !ok || len(x.MaxCounts) != size || x.CoefficientsUS == nil && name != "action" || x.CoefficientsUS != nil && !phaseCoefficients(x.CoefficientsUS, size) {
			return nil, fmt.Errorf("invalid queue service stage %s", name)
		}
		for _, n := range x.MaxCounts {
			if n < 0 || x.CoefficientsUS == nil && n != 0 {
				return nil, fmt.Errorf("invalid or unsupported queue stage coverage %s", name)
			}
		}
		x.CoefficientsUS = append([]float64(nil), x.CoefficientsUS...)
		x.MaxCounts = append([]int64(nil), x.MaxCounts...)
		stages[name] = x
	}
	p.Stages = stages
	if err := configureSpillService(c, &p); err != nil {
		return nil, err
	}
	return &queueServiceCost{profile: p}, nil
}

func (m *queueServiceCost) price(counts map[string][]int64, names ...string) (sim.DecisionCostEstimate, error) {
	total := 0.0
	for _, name := range names {
		x, ok := m.profile.Stages[name]
		work, known := counts[name]
		if !ok || !known || len(work) != len(x.MaxCounts) {
			return sim.DecisionCostEstimate{}, fmt.Errorf("missing queue work stage %s", name)
		}
		for i, n := range work {
			if n < 0 || n > x.MaxCounts[i] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("queue %s feature %d exceeds evaluated scope: %d", name, i, n)
			}
			if x.CoefficientsUS == nil {
				if n != 0 {
					return sim.DecisionCostEstimate{}, fmt.Errorf("queue action %s has no measured nonzero work", name)
				}
				continue
			}
			total += float64(n) * x.CoefficientsUS[i]
		}
	}
	if math.IsNaN(total) || math.IsInf(total, 0) || total >= float64(math.MaxInt64) {
		return sim.DecisionCostEstimate{}, fmt.Errorf("queue service overflow")
	}
	return sim.DecisionCostEstimate{ExtraUS: int64(math.Ceil(total)), Provenance: m.profile.Provenance + "; " + m.profile.Coverage}, nil
}

func (m *queueServiceCost) Estimate(v sim.DecisionView, p sim.DecisionPlan) (sim.DecisionCostEstimate, error) {
	counts, err := queueDecisionCounts(m.profile.Shape, v, p)
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	base, err := m.price(counts, "decision", "reservation_view", "estimate")
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	if m.profile.Spill != nil {
		work, err := SpillDecisionCounts(v, p, m.profile.Spill.Mode)
		if err != nil {
			return sim.DecisionCostEstimate{}, err
		}
		base, err = m.addSpillService(base, work, false)
		if err != nil {
			return sim.DecisionCostEstimate{}, err
		}
	}
	if extra := m.profile.ReestimateExtraUS; extra > 0 {
		if base.ExtraUS > math.MaxInt64-extra {
			return sim.DecisionCostEstimate{}, fmt.Errorf("queue budget reestimate service overflow")
		}
		base.ExtraUS += extra
		base.Provenance += "; budget reestimate additional service: " + m.profile.ReestimateProvenance
	}
	return base, nil
}

func (m *queueServiceCost) EstimateExecutionOutcome(v sim.DecisionView, p sim.DecisionPlan, o sim.ExecutionOutcome) (sim.DecisionCostEstimate, error) {
	if o.PreemptionCount != int64(len(o.Preemptions)) {
		return sim.DecisionCostEstimate{}, fmt.Errorf("queue service has an unprofiled preemption kind")
	}
	f := sim.DecisionFeedback{Version: v.Version, Status: "applied", Grants: o.Grants, Capacity: o.Capacity, Preemptions: o.Preemptions, PreemptionStorage: o.PreemptionStorage}
	counts, err := QueueExecutionCounts(v, o.Work, f, m.profile.Family == "budget_queues")
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	base, err := m.price(counts, "execution", "feedback", "action")
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	if m.profile.Spill != nil {
		work, err := SpillExecutionCounts(v, p, f, m.profile.Spill.MaxSourceBlocks)
		if err != nil {
			return sim.DecisionCostEstimate{}, err
		}
		return m.addSpillService(base, work, true)
	}
	if len(o.PreemptionStorage) != 0 {
		return sim.DecisionCostEstimate{}, fmt.Errorf("unprofiled storage outcome")
	}
	return base, nil
}

func (m *queueServiceCost) EstimateHostService(w sim.HostServiceWork) (sim.DecisionCostEstimate, error) {
	switch w.Stage {
	case "registration":
		if len(w.Requests) != 1 || len(w.Finishing) != 0 {
			return sim.DecisionCostEstimate{}, fmt.Errorf("invalid queue registration work")
		}
		return m.price(map[string][]int64{"registration": {1}}, "registration")
	case "completion":
		allowed := map[string]bool{}
		for _, id := range w.Requests {
			if allowed[id] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("duplicate returned queue request")
			}
			allowed[id] = true
		}
		for _, id := range w.Finishing {
			if !allowed[id] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("finishing queue request was not returned or is duplicated")
			}
			delete(allowed, id)
		}
		return m.price(map[string][]int64{"completion": {1, int64(len(w.Requests)), int64(len(w.Finishing))}}, "completion")
	default:
		return sim.DecisionCostEstimate{}, fmt.Errorf("unknown queue host work %s", w.Stage)
	}
}
