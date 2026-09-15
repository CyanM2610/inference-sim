package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

type EnginePhaseCurve struct {
	PreForward  []float64 `json:"pre_forward_us"`
	PostForward []float64 `json:"post_forward_us"`
	GPUReady    []float64 `json:"gpu_ready_us"`
	OutputReady []float64 `json:"output_ready_us"`
	Poll        []float64 `json:"poll_us"`
	Tail        []float64 `json:"tail_us"`
}

type EnginePhaseCostConfig struct {
	WorkerMetadata *WorkerMetadataCostConfig `json:"worker_metadata,omitempty"`
	Compute        EnginePhaseCurve          `json:"compute"`
	LoadPlanning   *EngineLoadPlanningCost   `json:"load_planning,omitempty"`
	// Optional measured specialization for batches with exactly one query token
	// per request. Applies to both prefill and decode; never selects by policy.
	SingleTokenCompute *EnginePhaseCurve   `json:"single_token_compute,omitempty"`
	Idle               kv.EngineStepTiming `json:"idle"`
	LoadSubmit         []float64           `json:"load_submit_us"`
	StoreSubmit        []float64           `json:"store_submit_us"`
	BlockBytes         int64               `json:"block_bytes"`
	Provenance         string              `json:"provenance"`
}

// EngineLoadPlanningCost models [one active step, new load jobs, new blocks].
// It must be measured above the existing phase baseline, not from total TTFT.
type EngineLoadPlanningCost struct {
	Coefficients []float64 `json:"coefficients_us"`
	MaxLoads     int64     `json:"max_loads"`
	MaxBlocks    int64     `json:"max_blocks"`
	Provenance   string    `json:"provenance"`
	Coverage     string    `json:"coverage"`
}

func (c EnginePhaseCostConfig) PredictLoadPlanning(loads, blocks int64) (int64, error) {
	p := c.LoadPlanning
	if p == nil || (loads == 0 && blocks == 0) {
		return 0, nil
	}
	if loads <= 0 || blocks < loads || loads > p.MaxLoads || blocks > p.MaxBlocks {
		return 0, fmt.Errorf("load planning outside profile: loads=%d blocks=%d", loads, blocks)
	}
	return phaseCost(p.Coefficients, []float64{1, float64(loads), float64(blocks)}), nil
}

func phaseCoefficients(v []float64, n int) bool {
	if len(v) != n {
		return false
	}
	for _, x := range v {
		if x < 0 || math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}
func (c EnginePhaseCostConfig) Validate() error {
	if c.WorkerMetadata != nil {
		if err := c.WorkerMetadata.Validate(); err != nil {
			return err
		}
	}
	if p := c.LoadPlanning; p != nil {
		if !phaseCoefficients(p.Coefficients, 3) || p.MaxLoads <= 0 || p.MaxBlocks < p.MaxLoads || p.Provenance == "" || p.Coverage == "" {
			return fmt.Errorf("load planning requires finite coefficients, bounds, provenance and coverage")
		}
	}
	if c.Provenance == "" || c.BlockBytes <= 0 || !phaseCoefficients(c.LoadSubmit, 2) || !phaseCoefficients(c.StoreSubmit, 2) {
		return fmt.Errorf("engine phases require provenance, block bytes and nonnegative host submission costs")
	}
	curves := []EnginePhaseCurve{c.Compute}
	if c.SingleTokenCompute != nil {
		curves = append(curves, *c.SingleTokenCompute)
	}
	for _, x := range curves {
		for _, curve := range [][]float64{x.PreForward, x.PostForward, x.GPUReady, x.OutputReady, x.Poll, x.Tail} {
			if !phaseCoefficients(curve, 4) {
				return fmt.Errorf("engine phase curves require four finite nonnegative coefficients")
			}
		}
		if x.Poll[0] <= 0 || x.Tail[0] <= 0 {
			return fmt.Errorf("engine phase curves require positive control costs")
		}
	}
	if c.Compute.Poll[0] <= 0 || c.Compute.Tail[0] <= 0 || c.LoadSubmit[0] <= 0 || c.StoreSubmit[0] <= 0 || c.Idle.PreForwardUS < 0 || c.Idle.PostForwardUS < c.Idle.PreForwardUS || c.Idle.GPUReadyUS != 0 || c.Idle.OutputReadyUS != 0 || c.Idle.PollUS <= 0 || c.Idle.TailUS <= 0 {
		return fmt.Errorf("engine phases require positive control/submit costs and an idle curve without GPU work")
	}
	return nil
}

func phaseCost(coefficients, features []float64) int64 {
	v := 0.0
	for i, x := range coefficients {
		v += x * features[i]
	}
	if math.IsNaN(v) || math.IsInf(v, 0) || v >= float64(math.MaxInt64/16) {
		panic("engine phase cost overflow")
	}
	return int64(math.Ceil(v))
}
func (c EnginePhaseCostConfig) PredictEngineStep(work []sim.BatchWork) kv.EngineStepTiming {
	if c.WorkerMetadata != nil && len(work) > 0 {
		panic("worker metadata profile requires explicit execution state")
	}
	if len(work) == 0 {
		return c.Idle
	}
	f := []float64{1, 0, 0, float64(len(work))}
	singleToken := true
	for _, w := range work {
		singleToken = singleToken && w.NewTokens == 1
		q := float64(w.NewTokens)
		f[1] += q
		f[2] += q * (float64(w.PrefixTokens) + (q+1)/2) / 1024
	}
	x := c.Compute
	if singleToken && c.SingleTokenCompute != nil {
		x = *c.SingleTokenCompute
	}
	pre := phaseCost(x.PreForward, f)
	post := max(pre, phaseCost(x.PostForward, f)) // fitted boundaries must respect causal ordering
	return kv.EngineStepTiming{PreForwardUS: pre, PostForwardUS: post, GPUReadyUS: phaseCost(x.GPUReady, f), OutputReadyUS: phaseCost(x.OutputReady, f), PollUS: phaseCost(x.Poll, f), TailUS: phaseCost(x.Tail, f)}
}
func (c EnginePhaseCostConfig) HostSubmitUS(direction string, blocks int64) int64 {
	coeff := c.StoreSubmit
	if direction == "h2d" {
		coeff = c.LoadSubmit
	} else if direction != "d2h" {
		panic("unsupported phase transfer direction")
	}
	return phaseCost(coeff, []float64{1, float64(blocks)})
}
