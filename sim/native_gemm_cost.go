package sim

import (
	"fmt"
	"math"
	"sort"
	"strconv"
)

const NativeGEMMCostScope = "warmed_isolated_rotating_weight_gemm_only"

// NativeGEMMProfile is a measured surface for fixed matrix families. It is not
// a forward-pass model: CPU submission, inter-kernel gaps and other operators
// are deliberately absent. Runtime/kernel source identity is checked by the
// experiment manifest, in addition to the geometry checks performed here.
type NativeGEMMProfile struct {
	Schema           int                        `json:"schema"`
	Role             string                     `json:"role"`
	Device           NativeGEMMDevice           `json:"device"`
	DType            string                     `json:"dtype"`
	KernelConfig     NativeGEMMKernelConfig     `json:"kernel_config"`
	MaximumQueryRows int                        `json:"maximum_query_rows"`
	Curves           map[string]NativeGEMMCurve `json:"curves"`
}

type NativeGEMMDevice struct {
	Name       string `json:"name"`
	SMCount    int    `json:"sm_count"`
	Capability [2]int `json:"capability"`
}

type NativeGEMMKernelConfig struct {
	BlockM    int `json:"BLOCK_SIZE_M"`
	BlockN    int `json:"BLOCK_SIZE_N"`
	BlockK    int `json:"BLOCK_SIZE_K"`
	GroupM    int `json:"GROUP_SIZE_M"`
	NumStages int `json:"num_stages"`
	NumWarps  int `json:"num_warps"`
}

type NativeGEMMCurve struct {
	K         int                    `json:"k"`
	N         int                    `json:"n"`
	Bias      bool                   `json:"bias"`
	RowGroups map[string][][]float64 `json:"row_groups"`
}

// NativeGEMMQuery makes warmed/isolated execution explicit. Declaring them true
// does not establish that a full-model execution has the measured cache/weight
// regime. That transfer of calibration needs a separate full-model experiment.
type NativeGEMMQuery struct {
	Role         string                 `json:"role"`
	M            int                    `json:"m"`
	K            int                    `json:"k"`
	N            int                    `json:"n"`
	Bias         bool                   `json:"bias"`
	AStride      [2]int                 `json:"a_stride"`
	BStride      [2]int                 `json:"b_stride"`
	Device       NativeGEMMDevice       `json:"device"`
	DType        string                 `json:"dtype"`
	KernelConfig NativeGEMMKernelConfig `json:"config"`
	Warmed       bool                   `json:"warmed"`
	Isolated     bool                   `json:"isolated"`
}

type NativeGEMMEstimate struct {
	GPUServiceUS    float64 `json:"gpu_service_us"`
	ClockDurationUS int64   `json:"clock_duration_us"`
	Scope           string  `json:"scope"`
}

// NativeGEMMEstimator owns a validated snapshot; later edits to the input
// profile cannot change the costs during a simulation.
type NativeGEMMEstimator struct {
	profile NativeGEMMProfile
}

func NewNativeGEMMEstimator(p NativeGEMMProfile) (*NativeGEMMEstimator, error) {
	c := p.KernelConfig
	if p.Schema != 1 || p.Role != "warm_native_gemm_within_row_tile_surface" || p.DType == "" || p.MaximumQueryRows <= 0 || len(p.Curves) == 0 || p.Device.Name == "" || p.Device.SMCount <= 0 || p.Device.Capability[0] <= 0 || p.Device.Capability[1] < 0 || c.BlockM <= 0 || c.BlockN <= 0 || c.BlockK <= 0 || c.GroupM <= 0 || c.NumStages <= 0 || c.NumWarps <= 0 {
		return nil, fmt.Errorf("invalid native GEMM profile identity or geometry")
	}
	snapshot := p
	snapshot.Curves = make(map[string]NativeGEMMCurve, len(p.Curves))
	maxGroup := (p.MaximumQueryRows-1)/c.BlockM + 1
	for role, curve := range p.Curves {
		if role == "" || curve.K <= 0 || curve.N <= 0 || len(curve.RowGroups) == 0 {
			return nil, fmt.Errorf("invalid native GEMM family %q", role)
		}
		owned := curve
		owned.RowGroups = make(map[string][][]float64, len(curve.RowGroups))
		for key, points := range curve.RowGroups {
			group, err := strconv.Atoi(key)
			if err != nil || group <= 0 || group > maxGroup || strconv.Itoa(group) != key || len(points) == 0 {
				return nil, fmt.Errorf("invalid native GEMM row group %q/%q", role, key)
			}
			maxOffset := c.BlockM
			if group == maxGroup {
				maxOffset = (p.MaximumQueryRows-1)%c.BlockM + 1
			}
			owned.RowGroups[key] = make([][]float64, len(points))
			previous := 0.0
			for i, point := range points {
				if len(point) != 2 || math.IsNaN(point[0]) || math.IsInf(point[0], 0) || point[0] != math.Trunc(point[0]) || point[0] <= previous || point[0] > float64(maxOffset) || math.IsNaN(point[1]) || math.IsInf(point[1], 0) || point[1] <= 0 || point[1] >= float64(math.MaxInt64) {
					return nil, fmt.Errorf("invalid native GEMM anchor %q/%q/%d", role, key, i)
				}
				previous = point[0]
				owned.RowGroups[key][i] = append([]float64(nil), point...)
			}
		}
		snapshot.Curves[role] = owned
	}
	return &NativeGEMMEstimator{profile: snapshot}, nil
}

// Estimate interpolates only inside one measured ceil(M/BM) group. Unsupported
// shapes and execution contexts return errors, never zero-cost fallbacks or
// silent extrapolation. ClockDurationUS rounds one primitive upward once; it
// does not describe exclusive resource occupancy or a whole forward envelope.
func (e *NativeGEMMEstimator) Estimate(q NativeGEMMQuery) (NativeGEMMEstimate, error) {
	if e == nil || e.profile.Schema == 0 {
		return NativeGEMMEstimate{}, fmt.Errorf("uninitialized native GEMM estimator")
	}
	p := e.profile
	if !q.Warmed || !q.Isolated || q.Device != p.Device || q.DType != p.DType || q.KernelConfig != p.KernelConfig {
		return NativeGEMMEstimate{}, fmt.Errorf("unprofiled native GEMM execution context")
	}
	curve, ok := p.Curves[q.Role]
	if !ok || q.M <= 0 || q.M > p.MaximumQueryRows || q.K != curve.K || q.N != curve.N || q.Bias != curve.Bias || q.AStride != [2]int{q.K, 1} || q.BStride != [2]int{1, q.K} {
		return NativeGEMMEstimate{}, fmt.Errorf("unprofiled native GEMM matrix or layout")
	}
	group, offset := (q.M-1)/p.KernelConfig.BlockM+1, float64((q.M-1)%p.KernelConfig.BlockM+1)
	points := curve.RowGroups[strconv.Itoa(group)]
	if len(points) == 0 || offset < points[0][0] || offset > points[len(points)-1][0] {
		return NativeGEMMEstimate{}, fmt.Errorf("unmeasured native GEMM row group or occupancy")
	}
	at := sort.Search(len(points), func(i int) bool { return points[i][0] >= offset })
	value := points[at][1]
	if points[at][0] != offset {
		left, right := points[at-1], points[at]
		value = left[1] + (right[1]-left[1])*(offset-left[0])/(right[0]-left[0])
	}
	return NativeGEMMEstimate{GPUServiceUS: value, ClockDurationUS: int64(math.Ceil(value)), Scope: NativeGEMMCostScope}, nil
}
