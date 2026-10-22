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
	Compute     EnginePhaseCurve    `json:"compute"`
	Idle        kv.EngineStepTiming `json:"idle"`
	LoadSubmit  []float64           `json:"load_submit_us"`
	StoreSubmit []float64           `json:"store_submit_us"`
	BlockBytes  int64               `json:"block_bytes"`
	Provenance  string              `json:"provenance"`
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
	if c.Provenance == "" || c.BlockBytes <= 0 || !phaseCoefficients(c.LoadSubmit, 2) || !phaseCoefficients(c.StoreSubmit, 2) {
		return fmt.Errorf("engine phases require provenance, block bytes and nonnegative host submission costs")
	}
	for _, curve := range [][]float64{c.Compute.PreForward, c.Compute.PostForward, c.Compute.GPUReady, c.Compute.OutputReady, c.Compute.Poll, c.Compute.Tail} {
		if !phaseCoefficients(curve, 4) {
			return fmt.Errorf("engine phase curves require four finite nonnegative coefficients")
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
	if len(work) == 0 {
		return c.Idle
	}
	f := []float64{1, 0, 0, float64(len(work))}
	for _, w := range work {
		q := float64(w.NewTokens)
		f[1] += q
		f[2] += q * (float64(w.PrefixTokens) + (q+1)/2) / 1024
	}
	x := c.Compute
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
