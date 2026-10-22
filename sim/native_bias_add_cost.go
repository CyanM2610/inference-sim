package sim

import (
	"fmt"
	"math"
	"sort"
)

const NativeBiasAddCostScope = "warmed_isolated_producer_local_bias_add_only"

// NativeBiasAddProfile describes the separate out-of-place addition following
// the actual unfused QKV GEMM. It does not cover a fused epilogue, an in-place
// operation, or a cold input loaded independently of the producer.
type NativeBiasAddProfile struct {
	Schema   int              `json:"schema"`
	Role     string           `json:"role"`
	DType    string           `json:"dtype"`
	Device   NativeGEMMDevice `json:"device"`
	N        int              `json:"n"`
	MinimumM int              `json:"minimum_m"`
	MaximumM int              `json:"maximum_m"`
	Points   [][]float64      `json:"points"`
}

type NativeBiasAddQuery struct {
	M             int              `json:"m"`
	N             int              `json:"n"`
	InputStride   [2]int           `json:"input_stride"`
	OutputStride  [2]int           `json:"output_stride"`
	BiasStride    [1]int           `json:"bias_stride"`
	DType         string           `json:"dtype"`
	Device        NativeGEMMDevice `json:"device"`
	Warmed        bool             `json:"warmed"`
	Isolated      bool             `json:"isolated"`
	ProducerLocal bool             `json:"producer_local"`
	InPlace       bool             `json:"in_place"`
}

type NativeBiasAddEstimate struct {
	GPUServiceUS    float64 `json:"gpu_service_us"`
	ClockDurationUS int64   `json:"clock_duration_us"`
	Scope           string  `json:"scope"`
}

// NativeBiasAddEstimator owns its calibration data. Source/runtime identity is
// additionally bound by the experiment manifest, as with NativeGEMMEstimator.
type NativeBiasAddEstimator struct {
	profile NativeBiasAddProfile
}

func NewNativeBiasAddEstimator(p NativeBiasAddProfile) (*NativeBiasAddEstimator, error) {
	if p.Schema != 1 || p.Role != "warm_native_producer_local_bias_add" || p.DType == "" || p.Device.Name == "" || p.Device.SMCount <= 0 || p.Device.Capability[0] <= 0 || p.Device.Capability[1] < 0 || p.N <= 0 || p.MinimumM <= 0 || p.MaximumM < p.MinimumM || len(p.Points) == 0 {
		return nil, fmt.Errorf("invalid native bias-add profile")
	}
	owned := p
	owned.Points = make([][]float64, len(p.Points))
	previous := 0.0
	for i, point := range p.Points {
		if len(point) != 2 || math.IsNaN(point[0]) || math.IsInf(point[0], 0) || point[0] != math.Trunc(point[0]) || point[0] <= previous || point[0] < float64(p.MinimumM) || point[0] > float64(p.MaximumM) || math.IsNaN(point[1]) || math.IsInf(point[1], 0) || point[1] <= 0 || point[1] >= float64(math.MaxInt64) {
			return nil, fmt.Errorf("invalid native bias-add anchor %d", i)
		}
		owned.Points[i] = append([]float64(nil), point...)
		previous = point[0]
	}
	if p.Points[0][0] != float64(p.MinimumM) || p.Points[len(p.Points)-1][0] != float64(p.MaximumM) {
		return nil, fmt.Errorf("native bias-add bounds are not measured")
	}
	return &NativeBiasAddEstimator{profile: owned}, nil
}

// Estimate returns one primitive's duration. Execution must separately account
// for its GEMM producer, CPU dispatch, allocation and other model operations.
// The estimate does not assign exclusive GPU occupancy or a forward latency.
func (e *NativeBiasAddEstimator) Estimate(q NativeBiasAddQuery) (NativeBiasAddEstimate, error) {
	if e == nil || e.profile.Schema == 0 {
		return NativeBiasAddEstimate{}, fmt.Errorf("uninitialized native bias-add estimator")
	}
	p := e.profile
	if !q.Warmed || !q.Isolated || !q.ProducerLocal || q.InPlace || q.Device != p.Device || q.DType != p.DType || q.M < p.MinimumM || q.M > p.MaximumM || q.N != p.N || q.InputStride != [2]int{p.N, 1} || q.OutputStride != [2]int{p.N, 1} || q.BiasStride != [1]int{1} {
		return NativeBiasAddEstimate{}, fmt.Errorf("unprofiled native bias-add context or geometry")
	}
	at := sort.Search(len(p.Points), func(i int) bool { return p.Points[i][0] >= float64(q.M) })
	value := p.Points[at][1]
	if p.Points[at][0] != float64(q.M) {
		left, right := p.Points[at-1], p.Points[at]
		value = left[1] + (right[1]-left[1])*(float64(q.M)-left[0])/(right[0]-left[0])
	}
	return NativeBiasAddEstimate{GPUServiceUS: value, ClockDurationUS: int64(math.Ceil(value)), Scope: NativeBiasAddCostScope}, nil
}
