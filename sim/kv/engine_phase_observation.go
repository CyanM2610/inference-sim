package kv

import "fmt"

// EnginePhaseObservation records the boundaries already used by the executor.
// GPUStartEstimateUS is a forward-ready/previous-stream-ready envelope boundary,
// not a measured kernel launch or an SM occupancy estimate. Empty control steps
// have no GPU work, even when their dependency frontier names earlier GPU work.
type EnginePhaseObservation struct {
	Instance           string   `json:"instance"`
	Requests           []string `json:"requests"`
	HasCompute         bool     `json:"has_compute"`
	StartUS            int64    `json:"start_us"`
	PlanningEndUS      int64    `json:"planning_end_us"`
	PreForwardUS       int64    `json:"pre_forward_us"`
	ForwardReadyUS     int64    `json:"forward_ready_us"`
	PostForwardUS      int64    `json:"post_forward_us"`
	LoadSubmitEndUS    int64    `json:"load_submit_end_us"`
	PollUS             int64    `json:"poll_us"`
	AdoptUS            int64    `json:"adopt_us"`
	PreviousGPUReadyUS int64    `json:"previous_gpu_ready_us"`
	GPUStartEstimateUS int64    `json:"gpu_start_estimate_us"`
	GPUReadyUS         int64    `json:"gpu_ready_us"`
	OutputReadyUS      int64    `json:"output_ready_us"`
}

// SetEnginePhaseObserver installs diagnostic recording before execution. It
// adds no events, predictions, dependencies, fees, or policy-visible state.
func (s *PeerCache) SetEnginePhaseObserver(observe func(EnginePhaseObservation)) error {
	p := s.fabric.phases
	if p == nil || p.started || p.active {
		return fmt.Errorf("engine phase observer requires configured unused phases")
	}
	p.observePhases = observe
	return nil
}
