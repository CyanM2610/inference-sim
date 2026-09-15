package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// QueueDecisionCounts describes only work visible before native dispatch. It
// does not price that work or use actual grants to predict a decision's cost.
func QueueDecisionCounts(c Config, v sim.DecisionView, p sim.DecisionPlan) (map[string][]int64, error) {
	return queueDecisionCounts(serviceShape(c), v, p)
}

func queueDecisionCounts(shape RestoreServiceShape, v sim.DecisionView, p sim.DecisionPlan) (map[string][]int64, error) {
	if v.Version != p.Version || v.BlockTokens <= 0 || !v.Capabilities.CapacityReservations {
		return nil, fmt.Errorf("queue work requires matching decision and reservation view")
	}
	w, err := restoreDecisionWork(shape, v)
	if err != nil {
		return nil, err
	}
	visible := int64(len(v.Waiting) + len(v.Running))
	var blocks int64
	for _, group := range [][]sim.DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			if r.InputTokens < 0 {
				return nil, fmt.Errorf("negative queue prompt length")
			}
			n := r.InputTokens / v.BlockTokens
			if r.InputTokens%v.BlockTokens != 0 {
				n++
			}
			if blocks > math.MaxInt64-n {
				return nil, fmt.Errorf("queue block count overflow")
			}
			blocks += n
		}
	}
	return map[string][]int64{
		"decision":         {1, visible, w["candidates"], int64(len(v.KV.PendingTransfers)), int64(len(p.TokenCaps)), int64(len(p.Restores))},
		"reservation_view": {1, visible, blocks},
		"estimate":         {1, w["compute_steps"], w["candidates"]},
	}, nil
}

// QueueExecutionCounts is evaluated after formation from this invocation's
// actual results. The quota flag identifies the native adapter, not a request
// preference. A control-only round needs an explicit known-empty work record.
func QueueExecutionCounts(v sim.DecisionView, w sim.BatchExecutionWork, f sim.DecisionFeedback, quota bool) (map[string][]int64, error) {
	if f.Version != v.Version || f.Status != "applied" {
		return nil, fmt.Errorf("queue execution requires this decision's actual applied feedback")
	}
	counts, err := WaitExecutionCounts(w, int64(len(f.Preemptions)))
	if err != nil {
		return nil, err
	}
	if !quota {
		counts[5] = 0 // plain pressure does not install the quota wrapper
	}
	visible := map[string]bool{}
	for _, group := range [][]sim.DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			visible[r.ID] = true
		}
	}
	seen := map[string]bool{}
	for _, g := range f.Grants {
		if !visible[g.Request] || seen[g.Request] || g.Tokens <= 0 {
			return nil, fmt.Errorf("invalid actual queue grant")
		}
		seen[g.Request] = true
	}
	grants := int64(len(f.Grants))
	return map[string][]int64{
		"execution": append(counts, grants),
		"feedback":  {1, int64(len(v.Waiting) + len(v.Running)), grants, int64(len(f.Capacity)), int64(len(f.Preemptions))},
		"action":    {int64(len(f.Preemptions))},
	}, nil
}
