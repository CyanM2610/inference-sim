package policylab

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

// All six coefficients predict complete phase offsets, not extra service.
// A missing SingleTokenWithNew is an explicit unmeasured execution branch.
type WorkerMetadataCostConfig struct {
	Compute            EnginePhaseCurve  `json:"compute"`
	SingleToken        EnginePhaseCurve  `json:"single_token_compute"`
	SingleTokenWithNew *EnginePhaseCurve `json:"single_token_with_new,omitempty"`
	MaxNewRequests     int64             `json:"max_new_requests"`
	Provenance         string            `json:"provenance"`
	Coverage           string            `json:"coverage"`
}

func (p WorkerMetadataCostConfig) Validate() error {
	if p.MaxNewRequests <= 0 || p.Provenance == "" || p.Coverage == "" {
		return fmt.Errorf("worker metadata requires count coverage and provenance")
	}
	curves := []EnginePhaseCurve{p.Compute, p.SingleToken}
	if p.SingleTokenWithNew != nil {
		curves = append(curves, *p.SingleTokenWithNew)
	}
	for _, x := range curves {
		for _, values := range [][]float64{x.PreForward, x.PostForward, x.GPUReady, x.OutputReady, x.Poll, x.Tail} {
			if !phaseCoefficients(values, 6) {
				return fmt.Errorf("worker metadata phase requires six finite nonnegative coefficients")
			}
		}
		if x.Poll[0] <= 0 || x.Tail[0] <= 0 {
			return fmt.Errorf("worker metadata control stages must be positive")
		}
	}
	return nil
}

func (c EnginePhaseCostConfig) WorkerMetadataEnabled() bool { return c.WorkerMetadata != nil }

func (c EnginePhaseCostConfig) PredictEngineStepWithMetadata(work []sim.BatchWork, fresh []bool) (kv.EngineStepTiming, error) {
	if c.WorkerMetadata == nil {
		return c.PredictEngineStep(work), nil
	}
	if len(work) != len(fresh) {
		return kv.EngineStepTiming{}, fmt.Errorf("missing per-request worker metadata")
	}
	if len(work) == 0 {
		return c.Idle, nil
	}
	p := c.WorkerMetadata
	f := []float64{1, 0, 0, float64(len(work)), 0, 0}
	single := true
	for i, w := range work {
		if w.NewTokens <= 0 || w.PrefixTokens < 0 {
			return kv.EngineStepTiming{}, fmt.Errorf("invalid worker compute shape")
		}
		q := float64(w.NewTokens)
		f[1] += q
		f[2] += q * (float64(w.PrefixTokens) + (q+1)/2) / 1024
		single = single && w.NewTokens == 1
		if fresh[i] {
			f[4] = 1
			f[5]++
		}
	}
	if f[5] > float64(p.MaxNewRequests) {
		return kv.EngineStepTiming{}, fmt.Errorf("worker-new count outside profile: %v", f[5])
	}
	x := p.Compute
	if single {
		x = p.SingleToken
		if f[5] > 0 {
			if p.SingleTokenWithNew == nil {
				return kv.EngineStepTiming{}, fmt.Errorf("single-token worker-new phase not measured")
			}
			x = *p.SingleTokenWithNew
		}
	}
	pre := phaseCost(x.PreForward, f)
	return kv.EngineStepTiming{PreForwardUS: pre, PostForwardUS: max(pre, phaseCost(x.PostForward, f)),
		GPUReadyUS: phaseCost(x.GPUReady, f), OutputReadyUS: phaseCost(x.OutputReady, f),
		PollUS: phaseCost(x.Poll, f), TailUS: phaseCost(x.Tail, f)}, nil
}
