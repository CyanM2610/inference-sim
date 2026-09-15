package sim

import (
	"fmt"
	"math"
)

// Costs are predictions from public work, not remaining physical service or
// a guarantee that the saved prefix survives until re-admission.
type DecisionSpillEstimate struct {
	Request          string `json:"request"`
	Pool             string `json:"pool"`
	Unavailable      string `json:"unavailable,omitempty"`
	CapacityPossible bool   `json:"capacity_possible"`
	StoreUS          int64  `json:"store_us"`
	BacklogUS        int64  `json:"backlog_us"`
	CopyUS           int64  `json:"copy_us"`
	HostUS           int64  `json:"host_us"`
}

func ValidateSpillTarget(r DecisionRequest, blockTokens int64, target DecisionSpillTarget) error {
	if blockTokens <= 0 || r.ComputedTokens < 0 || target.CompleteBlocks != r.ComputedTokens/blockTokens || target.TailTokens != r.ComputedTokens%blockTokens {
		return fmt.Errorf("spill inventory does not match completed progress")
	}
	left := target.CompleteBlocks
	for _, n := range []int64{target.ReadyBlocks, target.PendingBlocks, target.MissingBlocks} {
		if n < 0 || n > left {
			return fmt.Errorf("invalid spill block partition")
		}
		left -= n
	}
	if left != 0 || target.AvailableBlocks < 0 {
		return fmt.Errorf("invalid spill capacity inventory")
	}
	return nil
}

func validateSpillEstimates(v DecisionView, e DecisionEstimates) error {
	if e.SpillCoverage == "" && e.SpillProvenance == "" && len(e.Spills) == 0 {
		return nil
	}
	if !v.Capabilities.RequestSpill || e.SpillCoverage == "" || e.SpillProvenance == "" {
		return fmt.Errorf("spill estimates require capability/provenance/coverage")
	}
	running := map[string]DecisionRequest{}
	for _, r := range v.Running {
		if r.PrefillPreemptible {
			running[r.ID] = r
		}
	}
	pools := map[string]bool{}
	for _, p := range v.KV.Pools {
		pools[p.ID] = true
	}
	targets := map[[2]string]DecisionSpillTarget{}
	for _, request := range v.KV.Requests {
		for _, target := range request.SpillTargets {
			key := [2]string{request.ID, target.Pool}
			if _, duplicate := targets[key]; duplicate {
				return fmt.Errorf("duplicate spill target")
			}
			targets[key] = target
		}
	}
	seen := map[[2]string]bool{}
	for _, row := range e.Spills {
		key := [2]string{row.Request, row.Pool}
		r, exists := running[row.Request]
		if !exists || !pools[row.Pool] || seen[key] {
			return fmt.Errorf("invalid/repeated spill estimate identity")
		}
		seen[key] = true
		if row.Unavailable != "" {
			if row.StoreUS != 0 || row.BacklogUS != 0 || row.CopyUS != 0 || row.HostUS != 0 || row.CapacityPossible {
				return fmt.Errorf("unavailable spill has costs/capacity")
			}
			continue
		}
		target, ok := targets[key]
		if !ok {
			return fmt.Errorf("spill estimate lacks inventory")
		}
		if err := ValidateSpillTarget(r, v.BlockTokens, target); err != nil {
			return err
		}
		if row.CapacityPossible != (target.MissingBlocks <= target.AvailableBlocks) {
			return fmt.Errorf("spill capacity prediction contradicts inventory")
		}
		for _, cost := range []int64{row.StoreUS, row.BacklogUS, row.CopyUS, row.HostUS} {
			if cost < 0 || cost >= math.MaxInt64/16 {
				return fmt.Errorf("invalid spill estimate cost")
			}
		}
		if row.StoreUS != row.BacklogUS+row.CopyUS+row.HostUS {
			return fmt.Errorf("spill cost partition mismatch")
		}
	}
	if len(seen) != len(running)*len(pools) {
		return fmt.Errorf("missing running-request spill estimate")
	}
	return nil
}
