package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// PlacementPhaseCosts projects a single hypothetical reader through the same
// engine boundaries as execution. It never receives a Request or future trace.
// Unknown future batch partners, arrivals, and capacity pressure are outside
// this conditional model, not free operations in the actual simulator.
type PlacementPhaseCosts struct {
	model        EnginePhaseModel
	chunk        int64
	decisionUS   int64
	blockBytes   int64
	readLatency  int64
	readPerByte  float64
	writeLatency int64
	writePerByte float64
}

// A predictor may detach execution-only counters/reporters while retaining
// exactly the same coefficients. Hypothetical shapes are not executed work.
type PlacementPredictionModel interface {
	PlacementPredictionModel() EnginePhaseModel
}

type PlacementAccessCost struct {
	ComputeInitialGPUWaitUS int64 `json:"compute_initial_gpu_wait_us,omitempty"`
	TotalUS                 int64 `json:"total_us"`
	ComputeUS               int64 `json:"compute_us"`
	ComputedTokens          int64 `json:"computed_tokens"`
	ComputeSteps            int64 `json:"compute_steps"`
	LoadReadyUS             int64 `json:"load_ready_us"`
	LoadPlanningUS          int64 `json:"load_planning_us"`
	LoadSubmitUS            int64 `json:"load_submit_us"`
	LoadDataUS              int64 `json:"load_data_us"`
	LoadQueueUS             int64 `json:"load_queue_us"`
	LoadAdoptionWaitUS      int64 `json:"load_adoption_wait_us"`
	PendingStoreWaitUS      int64 `json:"pending_store_wait_us"`
	LoadedBlocks            int64 `json:"loaded_blocks"`
	LoadGroups              int64 `json:"load_groups"`
}

type PlacementLoadCost struct {
	ReadyUS        int64
	PlanningUS     int64
	SubmitUS       int64
	DataUS         int64
	QueueUS        int64
	AdoptionWaitUS int64
}

func newPlacementPhaseCosts(model EnginePhaseModel, chunk, decisionUS, bytes int64, read, write []PeerResource) (*PlacementPhaseCosts, error) {
	if model == nil || chunk <= 0 || decisionUS < 0 || bytes <= 0 || len(read) == 0 || len(write) == 0 {
		return nil, fmt.Errorf("placement costs require engine model, positive chunk/bytes and two paths")
	}
	if m, ok := model.(PlacementPredictionModel); ok {
		model = m.PlacementPredictionModel()
	}
	c := &PlacementPhaseCosts{model: model, chunk: chunk, decisionUS: decisionUS, blockBytes: bytes}
	path := func(resources []PeerResource) (int64, float64, error) {
		latency := int64(0)
		perByte := 0.0
		for _, r := range resources {
			if r.LatencyUS < 0 || r.BytesPerUS <= 0 || math.IsInf(r.BytesPerUS, 0) || math.IsNaN(r.BytesPerUS) {
				return 0, 0, fmt.Errorf("invalid placement path")
			}
			latency += r.LatencyUS
			perByte = max(perByte, 1/r.BytesPerUS)
		}
		return latency, perByte, nil
	}
	var err error
	if c.readLatency, c.readPerByte, err = path(read); err != nil {
		return nil, err
	}
	if c.writeLatency, c.writePerByte, err = path(write); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *PlacementPhaseCosts) timing(prefix, query int64, fresh bool) EngineStepTiming {
	work := []sim.BatchWork{{PrefixTokens: prefix, NewTokens: query}}
	if m, ok := c.model.(EngineWorkerMetadataModel); ok && m.WorkerMetadataEnabled() {
		t, err := m.PredictEngineStepWithMetadata(work, []bool{fresh})
		if err != nil {
			panic(err)
		}
		return t
	}
	return c.model.PredictEngineStep(work)
}

func (c *PlacementPhaseCosts) compute(prefix, query int64) PlacementAccessCost {
	return c.computeWithGPUWait(prefix, query, 0)
}

func (c *PlacementPhaseCosts) computeWithGPUWait(prefix, query, initialGPUReady int64) PlacementAccessCost {
	if prefix < 0 || query <= 0 || initialGPUReady < 0 {
		panic("invalid conditional compute shape")
	}
	r := PlacementAccessCost{ComputedTokens: query}
	gpuReady := initialGPUReady
	for query > 0 {
		q := min(c.chunk, query)
		t := c.timing(prefix, q, r.ComputeSteps == 0)
		base := r.TotalUS + c.decisionUS
		gpuBase := max(base, gpuReady)
		if r.ComputeSteps == 0 {
			r.ComputeInitialGPUWaitUS = gpuBase - base
		}
		gpuReady = gpuBase + t.GPUReadyUS
		r.TotalUS = max(base+t.PostForwardUS+t.PollUS, gpuBase+t.OutputReadyUS) + t.TailUS
		r.ComputeSteps++
		prefix += q
		query -= q
	}
	r.ComputeUS = r.TotalUS
	return r
}

func (c *PlacementPhaseCosts) dataUS(direction string, blocks int64) int64 {
	if blocks <= 0 {
		return 0
	}
	latency, rate := c.readLatency, c.readPerByte
	if direction == "d2h" {
		latency, rate = c.writeLatency, c.writePerByte
	}
	return max(1, latency+int64(math.Ceil(float64(blocks)*float64(c.blockBytes)*rate)))
}

// storeService is a vector operation cost. Its difference at n+k and n is the
// incremental current write cost; the fixed setup is not charged k times.
func (c *PlacementPhaseCosts) storeService(blocks int64) int64 {
	if blocks <= 0 {
		return 0
	}
	return c.model.HostSubmitUS("d2h", blocks) + c.dataUS("d2h", blocks)
}

func (c *PlacementPhaseCosts) load(blocks, queueUS, gpuWaitUS int64) PlacementLoadCost {
	if blocks <= 0 || queueUS < 0 || gpuWaitUS < 0 {
		panic("invalid conditional load shape")
	}
	r := PlacementLoadCost{SubmitUS: c.model.HostSubmitUS("h2d", blocks), DataUS: c.dataUS("h2d", blocks)}
	if m, ok := c.model.(EngineLoadPlanningModel); ok {
		var err error
		r.PlanningUS, err = m.PredictLoadPlanning(1, blocks)
		if err != nil {
			panic(err)
		}
	}
	t := c.model.PredictEngineStep(nil)
	submitEnd := c.decisionUS + r.PlanningUS + t.PostForwardUS + r.SubmitUS
	start := max(submitEnd, queueUS, gpuWaitUS)
	r.QueueUS = start - submitEnd
	physicalEnd := start + r.DataUS
	poll := submitEnd + t.PollUS
	// Missed completions are visible only at a later poll/adoption. Includes
	// one declared scheduler service per idle step, not a zero-time busy loop.
	period := c.decisionUS + t.PostForwardUS + t.PollUS + t.TailUS
	if period <= 0 {
		panic("nonpositive conditional idle step")
	}
	if physicalEnd > poll {
		poll += ((physicalEnd - poll) + period - 1) / period * period
	}
	r.ReadyUS = poll + t.TailUS
	r.AdoptionWaitUS = r.ReadyUS - physicalEnd
	return r
}
