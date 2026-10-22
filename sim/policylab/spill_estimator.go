package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

const SpillEstimateCoverage = "full_public_transfer_work_no_progress_credit_spill_host_action_and_interference_unvalidated"

func (m *restoreEstimator) addSpillEstimates(v sim.DecisionView, estimates *sim.DecisionEstimates) error {
	if !v.Capabilities.RequestSpill {
		return fmt.Errorf("spill estimator requires request spill capability")
	}
	estimates.SpillProvenance = m.phase.Provenance + "; configured STORE resources; full public backlog"
	estimates.SpillCoverage = SpillEstimateCoverage
	backlog, err := m.backlogForPathUS(v, m.access.WritePath)
	if err != nil {
		return err
	}
	states := map[string]sim.DecisionKVRequest{}
	for _, q := range v.KV.Requests {
		states[q.ID] = q
	}
	for _, r := range v.Running {
		if !r.PrefillPreemptible {
			continue
		}
		row := sim.DecisionSpillEstimate{Request: r.ID, Pool: m.access.Pool}
		var target *sim.DecisionSpillTarget
		for _, candidate := range states[r.ID].SpillTargets {
			if candidate.Pool == row.Pool {
				if target != nil {
					return fmt.Errorf("duplicate spill inventory target")
				}
				copy := candidate
				target = &copy
			}
		}
		if target == nil {
			row.Unavailable = "missing_spill_inventory"
		} else {
			if err := sim.ValidateSpillTarget(r, v.BlockTokens, *target); err != nil {
				return err
			}
			row.CapacityPossible = target.MissingBlocks <= target.AvailableBlocks
			if target.MissingBlocks > 0 || target.PendingBlocks > 0 {
				row.BacklogUS = backlog
				if target.PendingBlocks > 0 && len(v.KV.PendingTransfers) == 0 {
					return fmt.Errorf("pending spill copy has no public transfer")
				}
				idle := m.phase.Idle
				// Isolated submit/adopt allowance. A concurrent forward may impose
				// additional delay; this is not a proven completion-time bound.
				row.HostUS = estimateAdd(idle.PreForwardUS, estimateAdd(idle.PostForwardUS, estimateAdd(idle.PollUS, idle.TailUS)))
				if target.MissingBlocks > 0 {
					if m.phase.BlockBytes <= 0 || target.MissingBlocks > math.MaxInt64/m.phase.BlockBytes {
						return fmt.Errorf("spill bytes overflow")
					}
					row.HostUS = estimateAdd(row.HostUS, m.phase.HostSubmitUS("d2h", target.MissingBlocks))
					row.CopyUS = m.pathUS(m.access.WritePath, target.MissingBlocks*m.phase.BlockBytes)
				}
				row.StoreUS = estimateAdd(row.BacklogUS, estimateAdd(row.HostUS, row.CopyUS))
			}
		}
		estimates.Spills = append(estimates.Spills, row)
	}
	return nil
}
