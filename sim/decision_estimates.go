package sim

import (
	"fmt"
	"math"
)

// DecisionEstimator is a prediction-only extension. Inputs contain no runtime
// objects or future completion times. It cannot reserve or execute resources.
type DecisionEstimator interface {
	Estimate(DecisionView) (DecisionEstimates, error)
}

type DecisionEstimates struct {
	SpillProvenance string                    `json:"spill_provenance,omitempty"`
	SpillCoverage   string                    `json:"spill_coverage,omitempty"`
	Spills          []DecisionSpillEstimate   `json:"spills,omitempty"`
	Provenance      string                    `json:"provenance"`
	Coverage        string                    `json:"coverage"`
	Requests        []DecisionRequestEstimate `json:"requests"`
}

type DecisionRequestEstimate struct {
	Request     string                    `json:"request"`
	Unavailable string                    `json:"unavailable,omitempty"`
	Choices     []DecisionRestoreEstimate `json:"choices,omitempty"`
}

type DecisionRestoreEstimate struct {
	MaxPrefixBlocks  int64 `json:"max_prefix_blocks"`
	LoadUS           int64 `json:"load_us"`
	ComputeUS        int64 `json:"compute_us"`
	MaxLoadComputeUS int64 `json:"max_load_compute_us"`
}

func cloneDecisionEstimates(e DecisionEstimates) DecisionEstimates {
	e.Spills = append([]DecisionSpillEstimate(nil), e.Spills...)
	e.Requests = append([]DecisionRequestEstimate(nil), e.Requests...)
	for i := range e.Requests {
		e.Requests[i].Choices = append([]DecisionRestoreEstimate(nil), e.Requests[i].Choices...)
	}
	return e
}

func (s *Simulator) SetDecisionEstimator(model DecisionEstimator) error {
	if s.decision == nil || model == nil || s.stepCount != 0 {
		return fmt.Errorf("decision estimator requires an installed policy before execution")
	}
	s.decision.estimator = model
	return nil
}

func ValidateDecisionEstimates(v DecisionView, e DecisionEstimates) error {
	if err := validateSpillEstimates(v, e); err != nil {
		return err
	}
	if e.Provenance == "" || e.Coverage == "" {
		return fmt.Errorf("decision estimates require provenance and coverage")
	}
	waiting := map[string]bool{}
	kv := map[string]DecisionKVRequest{}
	for _, r := range v.Waiting {
		if r.ComputedTokens < max(r.InputTokens, r.RecomputeUntilTokens) {
			waiting[r.ID] = true
		}
	}
	for _, r := range v.KV.Requests {
		kv[r.ID] = r
	}
	seen := map[string]bool{}
	for _, row := range e.Requests {
		if !waiting[row.Request] || seen[row.Request] {
			return fmt.Errorf("unknown or repeated estimate for %q", row.Request)
		}
		seen[row.Request] = true
		if row.Unavailable != "" {
			if len(row.Choices) != 0 {
				return fmt.Errorf("unavailable estimate has costs")
			}
			continue
		}
		state, ok := kv[row.Request]
		if !ok || !v.KV.PrefixStateKnown || len(row.Choices) == 0 {
			return fmt.Errorf("missing prefix/cost data for %q", row.Request)
		}
		previous := int64(-1)
		for _, c := range row.Choices {
			if c.MaxPrefixBlocks <= previous || c.MaxPrefixBlocks < state.LocalPrefixBlocks || c.MaxPrefixBlocks > state.RecoverablePrefixBlocks {
				return fmt.Errorf("invalid restore candidate for %q", row.Request)
			}
			for _, cost := range []int64{c.LoadUS, c.ComputeUS, c.MaxLoadComputeUS} {
				if cost < 0 || cost >= math.MaxInt64/16 {
					return fmt.Errorf("invalid estimated cost for %q", row.Request)
				}
			}
			previous = c.MaxPrefixBlocks
		}
		if row.Choices[0].MaxPrefixBlocks != state.LocalPrefixBlocks || row.Choices[0].LoadUS != 0 || previous != state.RecoverablePrefixBlocks {
			return fmt.Errorf("estimate must include local-only and full available prefix")
		}
	}
	if len(seen) != len(waiting) {
		return fmt.Errorf("missing waiting-request estimate")
	}
	return nil
}
