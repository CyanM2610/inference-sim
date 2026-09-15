package policylab

import (
	"fmt"
	"math"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

var decodeFillPrepareNames = [...]string{"fixed", "visible", "waiting", "pending", "input_blocks", "ready_decode", "promotions", "holds"}
var decodeFillExecutionNames = [...]string{"fixed", "visible", "waiting", "grants", "new_workers", "holds"}

type DecodeFillServiceCurve struct {
	CoefficientsNS []int64 `json:"coefficients_ns"`
	MaxFeatures    []int64 `json:"max_features"`
}

// This prices only the measured controller envelopes, not all engine/driver CPU.
type DecodeFillServiceCostConfig struct {
	Policy             string                 `json:"policy"`
	Prepare            DecodeFillServiceCurve `json:"prepare"`
	Execution          DecodeFillServiceCurve `json:"execution"`
	Shape              RestoreServiceShape    `json:"shape"`
	HBMBlocks          int64                  `json:"hbm_blocks"`
	PoolCapacityBlocks int64                  `json:"pool_capacity_blocks"`
	BlockBytes         int64                  `json:"block_bytes"`
	MaxInputTokens     int64                  `json:"max_input_tokens"`
	MaxOutputTokens    int                    `json:"max_output_tokens"`
	MaxPromotionBlocks int64                  `json:"max_promotion_blocks"`
	Provenance         string                 `json:"provenance"`
	Coverage           string                 `json:"coverage"`
}

type decodeFillServiceCost struct{ profile DecodeFillServiceCostConfig }

func newDecodeFillServiceCost(c Config) (*decodeFillServiceCost, error) {
	d := c.DecisionPolicy
	if d == nil || d.DecodeFillServiceCost == nil || d.BalancedBatch == nil || !d.PrefillDeferrals || !c.PromotionControl || c.PromotionRetention != "request" ||
		!c.RestoreControl || c.EnginePhases == nil || c.EnginePhases.WorkerMetadata == nil || c.Native != nil || len(c.Instances) != 1 || len(c.Pools) != 1 {
		return nil, fmt.Errorf("decode-fill service requires the measured balanced/promotion/deferral worker-state path")
	}
	if d.ExtraCost != nil || d.QueueServiceCost != nil || d.RestoreServiceCost != nil || d.WaitDecisionCost != nil || d.ExecutionCost != nil ||
		d.WaitExecutionCost != nil || d.HostServiceCost != nil || d.WaitHostServiceCost != nil || d.WaitPostStepCost != nil ||
		d.PrefillWaits || d.PrefillPreemption || d.CapacityPreemption || d.RequestSpill || d.ProducerWait != nil || d.ProducerPrefixes ||
		d.PrefixSharing || d.PrefillWaitProbe != nil || d.PrefillDeferralProbe != nil || c.BatchPrefixReuse || c.DecisionEstimates != nil ||
		d.CapacityReservationView || c.DecodeCapacityReservation || d.BalancedBatch.BundleHits || d.BalancedBatch.ReadyComputeBudget {
		return nil, fmt.Errorf("decode-fill service cannot reuse another controller profile or unmeasured policy/observer")
	}
	p := *d.DecodeFillServiceCost
	want := "balanced"
	if d.BalancedBatch.DecodeFill {
		want = "decode_fill"
	}
	if p.Policy != want || p.Shape != serviceShape(c) || p.HBMBlocks != c.Instances[0].HBMBlocks || p.PoolCapacityBlocks != c.Pools[0].CapacityBlocks ||
		p.BlockBytes != c.EnginePhases.BlockBytes || p.MaxInputTokens <= 0 || p.MaxOutputTokens <= 0 || p.MaxPromotionBlocks < 0 ||
		strings.TrimSpace(p.Provenance) == "" || strings.TrimSpace(p.Coverage) == "" {
		return nil, fmt.Errorf("decode-fill service policy/geometry/envelope/provenance mismatch")
	}
	for _, r := range c.Requests {
		if int64(len(r.Input)) > p.MaxInputTokens || r.MaxOutputTokens <= 0 || r.MaxOutputTokens > p.MaxOutputTokens {
			return nil, fmt.Errorf("decode-fill service request exceeds observed scope")
		}
	}
	curves := []*DecodeFillServiceCurve{&p.Prepare, &p.Execution}
	for i, curve := range curves {
		n := len(decodeFillPrepareNames)
		if i == 1 {
			n = len(decodeFillExecutionNames)
		}
		if len(curve.CoefficientsNS) != n || len(curve.MaxFeatures) != n || curve.MaxFeatures[0] != 1 {
			return nil, fmt.Errorf("invalid decode-fill service curve")
		}
		for j, rate := range curve.CoefficientsNS {
			if rate < 0 || curve.MaxFeatures[j] < 0 {
				return nil, fmt.Errorf("negative decode-fill service rate/envelope")
			}
		}
		if _, err := priceDecodeFillCurve(*curve, curve.MaxFeatures); err != nil {
			return nil, err
		}
		curve.CoefficientsNS = append([]int64(nil), curve.CoefficientsNS...)
		curve.MaxFeatures = append([]int64(nil), curve.MaxFeatures...)
	}
	return &decodeFillServiceCost{profile: p}, nil
}

// Prepare counts use only the current view/plan. Execution additionally uses
// actual grants, never candidate caps as a substitute for dispatched work.
func DecodeFillServiceFeatures(v sim.DecisionView, p sim.DecisionPlan, grants []sim.DecisionGrant) ([]int64, []int64, error) {
	if v.BlockTokens <= 0 || !v.KV.WorkerStateKnown {
		return nil, nil, fmt.Errorf("decode-fill service features require explicit worker residency")
	}
	states := map[string]sim.DecisionKVRequest{}
	for _, s := range v.KV.Requests {
		states[s.ID] = s
	}
	var blocks, ready int64
	for _, group := range [][]sim.DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			blocks += r.InputTokens / v.BlockTokens
		}
	}
	for _, r := range v.Running {
		s, ok := states[r.ID]
		if !ok {
			return nil, nil, fmt.Errorf("missing running KV service state")
		}
		if r.EmittedTokens > 0 && r.ComputedTokens >= max(r.InputTokens, r.RecomputeUntilTokens) && !s.TransferPending && !s.Deferred {
			ready++
		}
	}
	visible, waiting, holds := int64(len(v.Waiting)+len(v.Running)), int64(len(v.Waiting)), int64(len(p.PrefillDeferrals))
	prepare := []int64{1, visible, waiting, int64(len(v.KV.PendingTransfers)), blocks, ready, int64(len(p.Promotions)), holds}
	seen := map[string]bool{}
	var fresh int64
	for _, g := range grants {
		s, ok := states[g.Request]
		if !ok || seen[g.Request] || g.Tokens <= 0 {
			return nil, nil, fmt.Errorf("invalid actual grants for decode-fill service")
		}
		seen[g.Request] = true
		if !s.WorkerResident {
			fresh++
		}
	}
	return prepare, []int64{1, visible, waiting, int64(len(grants)), fresh, holds}, nil
}

func priceDecodeFillCurve(c DecodeFillServiceCurve, f []int64) (int64, error) {
	if len(f) != len(c.CoefficientsNS) || len(f) != len(c.MaxFeatures) {
		return 0, fmt.Errorf("decode-fill service feature width mismatch")
	}
	var ns int64
	for i, n := range f {
		if n < 0 || n > c.MaxFeatures[i] {
			return 0, fmt.Errorf("decode-fill service feature %d exceeds measured scope: %d > %d", i, n, c.MaxFeatures[i])
		}
		if c.CoefficientsNS[i] < 0 || n > 0 && c.CoefficientsNS[i] > (math.MaxInt64-999-ns)/n {
			return 0, fmt.Errorf("decode-fill service nanosecond sum overflows")
		}
		ns += n * c.CoefficientsNS[i]
	}
	return (ns + 999) / 1000, nil
}

func (c *decodeFillServiceCost) check(v sim.DecisionView, p sim.DecisionPlan) error {
	if !v.Capabilities.PrefillDeferrals || !v.Capabilities.Promotions || v.Capabilities.PromotionRetention != "request" || !v.KV.TransfersKnown ||
		!v.KV.WorkerStateKnown || v.KV.PrefixSharingKnown || v.Capabilities.CostEstimates || v.Capabilities.PrefillWaits || v.Capabilities.PrefillPreemption ||
		v.Capabilities.CapacityPreemption || v.Capabilities.PrefixProducers {
		return fmt.Errorf("unmeasured decode-fill controller capability combination")
	}
	// Native views do not expose pool occupancy; only configured capacity was
	// measured and checked at construction. Do not treat missing pools as empty.
	var promoted int64
	for _, a := range p.Promotions {
		promoted += a.MaxPrefixBlocks
	}
	if promoted > c.profile.MaxPromotionBlocks {
		return fmt.Errorf("decode-fill promotion work exceeds measured envelope")
	}
	for _, group := range [][]sim.DecisionRequest{v.Waiting, v.Running} {
		for _, r := range group {
			if r.InputTokens > c.profile.MaxInputTokens || r.ClientOutputLimit <= 0 || r.ClientOutputLimit > c.profile.MaxOutputTokens || r.RecomputeUntilTokens > r.InputTokens {
				return fmt.Errorf("unmeasured request history in decode-fill service")
			}
		}
	}
	return nil
}

func (c *decodeFillServiceCost) Estimate(v sim.DecisionView, p sim.DecisionPlan) (sim.DecisionCostEstimate, error) {
	if err := c.check(v, p); err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	f, _, err := DecodeFillServiceFeatures(v, p, nil)
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	us, err := priceDecodeFillCurve(c.profile.Prepare, f)
	return sim.DecisionCostEstimate{ExtraUS: us, Provenance: c.profile.Provenance + "; prepare excludes engine-owned native work"}, err
}

func (c *decodeFillServiceCost) EstimateExecutionOutcome(v sim.DecisionView, p sim.DecisionPlan, o sim.ExecutionOutcome) (sim.DecisionCostEstimate, error) {
	if err := c.check(v, p); err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	_, f, err := DecodeFillServiceFeatures(v, p, o.Grants)
	if err != nil {
		return sim.DecisionCostEstimate{}, err
	}
	us, err := priceDecodeFillCurve(c.profile.Execution, f)
	return sim.DecisionCostEstimate{ExtraUS: us, Provenance: c.profile.Provenance + "; actual dispatch/feedback before compute"}, err
}
