package policylab

import (
	"fmt"
	"math"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

// SpillServiceCostConfig prices exclusive additions to the eight base queue
// stages. Native STORE submit, DMA and flush remain owned by engine phases.
// NativeBoundaryExtraUS is a declared per-actual-preemption sensitivity allowance
// for residual metadata/worker bookkeeping, never their full parent intervals.
// An explicit zero selects the lower sensitivity endpoint, not a measured zero.
type SpillServiceCostConfig struct {
	profileReporter
	Mode                     string                       `json:"mode"`
	Stages                   map[string]QueueServiceCurve `json:"stages"`
	MaxRequests              int                          `json:"max_requests"`
	MaxSourceBlocks          int64                        `json:"max_source_blocks"`
	NativeBoundaryExtraUS    *int64                       `json:"native_boundary_extra_us"`
	NativeBoundaryProvenance string                       `json:"native_boundary_provenance"`
	Provenance               string                       `json:"provenance"`
	Coverage                 string                       `json:"coverage"`
}

func spillStageSizes(mode string) map[string]int {
	return map[string]int{"spill_view": 2, "spill_estimate": 3, "spill_plan": 2,
		"spill_policy_" + mode: 2, "spill_prepare": 2, "spill_store": 1}
}

func configureSpillService(c Config, p *QueueServiceCostConfig) error {
	d := c.DecisionPolicy
	if p.Spill == nil {
		if d.RequestSpill {
			return fmt.Errorf("request spill requires its exclusive service extension")
		}
		return nil
	}
	x := *p.Spill
	x.profileReporter = c.profile("spill_service", x.Provenance)
	if !d.RequestSpill || d.PreemptionStorage == nil || d.PreemptionStorage.Mode != x.Mode ||
		(x.Mode != "spill" && x.Mode != "recompute" && x.Mode != "budget") ||
		c.DecisionEstimates == nil || !c.DecisionEstimates.Spill || d.BudgetRestore.ReestimateOnDecodeDrop ||
		c.Mechanisms == nil || c.Mechanisms.BackgroundStoreMode != "on_preemption" || c.Mechanisms.BackgroundStorePool != "dram" ||
		len(c.Pools) != 1 || c.Pools[0].ID != "dram" || c.BlockTokens != 16 ||
		x.MaxRequests <= 0 || x.MaxSourceBlocks <= 0 ||
		x.NativeBoundaryExtraUS == nil || *x.NativeBoundaryExtraUS < 0 || strings.TrimSpace(x.NativeBoundaryProvenance) == "" ||
		strings.TrimSpace(x.Provenance) == "" || strings.TrimSpace(x.Coverage) == "" {
		return fmt.Errorf("spill service requires a declared on-preemption DRAM policy, estimates, scope and residual allowance")
	}
	x.above("requests", int64(len(c.Requests)), int64(x.MaxRequests))
	if p.Stages["action"].CoefficientsUS == nil {
		return fmt.Errorf("spill service requires a measured native preempt action in the base profile")
	}
	sizes := spillStageSizes(x.Mode)
	if len(x.Stages) != len(sizes) {
		return fmt.Errorf("spill service requires six disjoint stages for the selected mode")
	}
	stages := make(map[string]QueueServiceCurve, len(sizes))
	for name, n := range sizes {
		curve, ok := x.Stages[name]
		if !ok || !phaseCoefficients(curve.CoefficientsUS, n) || len(curve.MaxCounts) != n {
			return fmt.Errorf("invalid spill stage %s", name)
		}
		for _, count := range curve.MaxCounts {
			if count < 0 {
				return fmt.Errorf("negative spill scope %s", name)
			}
		}
		curve.CoefficientsUS = append([]float64(nil), curve.CoefficientsUS...)
		curve.MaxCounts = append([]int64(nil), curve.MaxCounts...)
		stages[name] = curve
	}
	x.Stages = stages
	allowance := *x.NativeBoundaryExtraUS
	x.NativeBoundaryExtraUS = &allowance
	p.Spill = &x
	return nil
}

// SpillDecisionCounts uses only the current snapshot and proposed plan. The
// policy feature basis collapses target/action columns only for the observed
// one-target-per-budget-choice relation; violations must not be priced as zero.
func SpillDecisionCounts(v sim.DecisionView, p sim.DecisionPlan, mode string) (map[string][]int64, error) {
	if v.Version != p.Version || !v.Capabilities.RequestSpill || (mode != "budget" && mode != "spill" && mode != "recompute") {
		return nil, fmt.Errorf("spill work requires a matching enabled decision and storage mode")
	}
	running := map[string]sim.DecisionRequest{}
	for _, r := range v.Running {
		running[r.ID] = r
	}
	var targets, blocks int64
	seen := map[string]bool{}
	for _, q := range v.KV.Requests {
		for _, t := range q.SpillTargets {
			r, ok := running[q.ID]
			if !ok || seen[q.ID] || t.Pool != "dram" {
				return nil, fmt.Errorf("unsupported spill target identity")
			}
			if err := sim.ValidateSpillTarget(r, v.BlockTokens, t); err != nil {
				return nil, err
			}
			seen[q.ID] = true
			if blocks > math.MaxInt64-t.CompleteBlocks {
				return nil, fmt.Errorf("spill inventory overflow")
			}
			targets++
			blocks += t.CompleteBlocks
		}
	}
	choices := int64(len(p.PreemptionStorage))
	if mode == "budget" && targets != choices {
		return nil, fmt.Errorf("spill budget target/action basis is outside measured coverage")
	}
	selected := map[string]bool{}
	for _, a := range p.PreemptionStorage {
		if !seen[a.Request] || selected[a.Request] || a.Mode != "recompute" && a.Mode != "spill" ||
			mode != "budget" && a.Mode != mode || a.Mode == "spill" && a.Pool != "dram" || a.Mode == "recompute" && a.Pool != "" {
			return nil, fmt.Errorf("unsupported spill storage choice")
		}
		selected[a.Request] = true
	}
	return map[string][]int64{"spill_view": {1, blocks}, "spill_estimate": {1, targets, int64(len(v.KV.PendingTransfers))},
		"spill_plan": {1, choices}, "spill_policy_" + mode: {1, choices}}, nil
}

// SpillExecutionCounts charges preparation only for actual victims. The current
// component data cover fresh complete prefixes, successful full STORE or explicit
// recompute. Partial/reused/pending/capacity-fallback actions need new coverage.
func SpillExecutionCounts(v sim.DecisionView, p sim.DecisionPlan, f sim.DecisionFeedback, maxSourceBlocks int64) (map[string][]int64, error) {
	return spillExecutionCounts(v, p, f, maxSourceBlocks, profileReporter{component: "spill_service"})
}

func spillExecutionCounts(v sim.DecisionView, p sim.DecisionPlan, f sim.DecisionFeedback, maxSourceBlocks int64, report profileReporter) (map[string][]int64, error) {
	if v.Version != p.Version || f.Version != v.Version || f.Status != "applied" || !v.Capabilities.RequestSpill ||
		len(f.Preemptions) != len(f.PreemptionStorage) || maxSourceBlocks <= 0 {
		return nil, fmt.Errorf("spill service requires complete actual preemption feedback")
	}
	actions := map[string]sim.DecisionPreemptionStorage{}
	for _, a := range p.PreemptionStorage {
		if _, ok := actions[a.Request]; ok {
			return nil, fmt.Errorf("duplicate planned spill choice")
		}
		actions[a.Request] = a
	}
	victims := map[string]sim.DecisionPreemptionOutcome{}
	for _, o := range f.Preemptions {
		if _, ok := victims[o.Request]; ok {
			return nil, fmt.Errorf("duplicate actual spill victim")
		}
		if o.Status != "requeued" {
			return nil, fmt.Errorf("unsupported actual spill preemption")
		}
		victims[o.Request] = o
	}
	var stores, recomputes int64
	for _, o := range f.PreemptionStorage {
		a, planned := actions[o.Request]
		victim, actual := victims[o.Request]
		if !planned || !actual {
			return nil, fmt.Errorf("storage outcome is not a unique actual planned victim")
		}
		delete(victims, o.Request)
		var target *sim.DecisionSpillTarget
		for _, q := range v.KV.Requests {
			if q.ID == o.Request {
				for _, t := range q.SpillTargets {
					if target != nil || t.Pool != "dram" {
						return nil, fmt.Errorf("ambiguous actual spill target")
					}
					copy := t
					target = &copy
				}
			}
		}
		if target == nil || target.CompleteBlocks <= 0 ||
			target.MissingBlocks != target.CompleteBlocks || target.ReadyBlocks != 0 || target.PendingBlocks != 0 ||
			v.BlockTokens <= 0 || victim.ComputedTokensBefore/v.BlockTokens != target.CompleteBlocks ||
			victim.ComputedTokensBefore%v.BlockTokens != target.TailTokens {
			return nil, fmt.Errorf("actual spill source is outside evaluated fresh-prefix scope")
		}
		report.above("source_blocks", target.CompleteBlocks, maxSourceBlocks)
		switch a.Mode {
		case "spill":
			if a.Pool != "dram" || o.Pool != a.Pool || o.Status != "spill_pending" || o.NewBlocks != target.CompleteBlocks ||
				o.ReadyBlocks != 0 || o.PendingBlocks != 0 || o.TailTokens != target.TailTokens {
				return nil, fmt.Errorf("unmeasured partial, reused or failed spill action")
			}
			stores++
		case "recompute":
			if a.Pool != "" || o.Pool != "" || o.Status != "recompute" || o.NewBlocks != 0 || o.ReadyBlocks != 0 ||
				o.PendingBlocks != 0 || o.TailTokens != victim.ComputedTokensBefore {
				return nil, fmt.Errorf("invalid actual recompute outcome")
			}
			recomputes++
		default:
			return nil, fmt.Errorf("unknown actual storage mode")
		}
	}
	return map[string][]int64{"spill_prepare": {stores, recomputes}, "spill_store": {stores}}, nil
}

func (m *queueServiceCost) addSpillService(base sim.DecisionCostEstimate, counts map[string][]int64, execution bool) (sim.DecisionCostEstimate, error) {
	p := m.profile.Spill
	names := []string{"spill_view", "spill_estimate", "spill_plan", "spill_policy_" + p.Mode}
	if execution {
		names = []string{"spill_prepare", "spill_store"}
	}
	total := 0.0
	for _, name := range names {
		curve := p.Stages[name]
		work, ok := counts[name]
		if !ok || len(work) != len(curve.MaxCounts) {
			return sim.DecisionCostEstimate{}, fmt.Errorf("missing spill work %s", name)
		}
		for i, n := range work {
			if n < 0 {
				return sim.DecisionCostEstimate{}, fmt.Errorf("negative spill %s feature %d: %d", name, i, n)
			}
			p.above(fmt.Sprintf("%s.feature_%d", name, i), n, curve.MaxCounts[i])
			total += float64(n) * curve.CoefficientsUS[i]
		}
	}
	if execution {
		for _, n := range counts["spill_prepare"] {
			total += float64(n) * float64(*p.NativeBoundaryExtraUS)
		}
	}
	if math.IsNaN(total) || math.IsInf(total, 0) || total >= float64(math.MaxInt64) {
		return sim.DecisionCostEstimate{}, fmt.Errorf("spill service overflow")
	}
	extra := int64(math.Ceil(total))
	if base.ExtraUS > math.MaxInt64-extra {
		return sim.DecisionCostEstimate{}, fmt.Errorf("combined spill service overflow")
	}
	base.ExtraUS += extra
	base.Provenance += "; exclusive spill: " + p.Provenance + "; " + p.Coverage + "; native boundary allowance: " + p.NativeBoundaryProvenance
	return base, nil
}
