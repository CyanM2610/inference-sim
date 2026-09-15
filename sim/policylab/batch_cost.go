package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

// BatchCostConfig describes a hardware/engine-specific primitive cost model.
// Coefficients must be fitted to independent batch observations, never policy
// TTFTs. This boundary includes the synchronous engine step (CPU + GPU).
type BatchCostConfig struct {
	CoefficientsUS []float64 `json:"coefficients_us"`
	EnqueueUS      int64     `json:"enqueue_us"`
	Provenance     string    `json:"provenance"`
}

func (c BatchCostConfig) Validate() error {
	if len(c.CoefficientsUS) != 4 || c.EnqueueUS < 0 || c.Provenance == "" {
		return fmt.Errorf("batch cost needs four nonnegative coefficients, enqueue time, and provenance")
	}
	for _, v := range c.CoefficientsUS {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("invalid batch cost coefficient")
		}
	}
	return nil
}

type batchCostLatency struct {
	store   *kv.PeerCache
	config  BatchCostConfig
	observe func(BatchCostObservation)
}

// BatchCostObservation is diagnostic input to the existing cost model. It does
// not expose request output budgets or mutate simulation state.
type BatchCostObservation struct {
	StartUS    int64              `json:"start_us"`
	DurationUS int64              `json:"duration_us"`
	Scheduled  []BatchCostRequest `json:"scheduled"`
}
type BatchCostRequest struct {
	Request string `json:"request"`
	Prefix  int64  `json:"prefix"`
	Query   int64  `json:"query"`
}

func (m *batchCostLatency) StepTime(batch []*sim.Request) int64 {
	// Features: intercept, total query tokens, attention pairs / 1024,
	// and scheduled sequence count. Context is the prefix BEFORE this batch.
	features := [4]float64{1, 0, 0, 0}
	var observation BatchCostObservation
	for _, r := range batch {
		if r.NumNewTokens <= 0 {
			continue
		}
		prefix := r.ProgressIndex
		if prefix == 0 {
			prefix = int64(len(m.store.GetCachedBlocks(r.FullInputTokens()))) * m.store.BlockSize()
		}
		query := float64(r.NumNewTokens)
		features[1] += query
		features[2] += query * (float64(prefix) + (query+1)/2) / 1024
		features[3]++
		if m.observe != nil {
			observation.Scheduled = append(observation.Scheduled, BatchCostRequest{Request: r.ID, Prefix: prefix, Query: int64(r.NumNewTokens)})
		}
	}
	value := 0.0
	for i, c := range m.config.CoefficientsUS {
		value += c * features[i]
	}
	if math.IsNaN(value) || math.IsInf(value, 0) || value >= float64(math.MaxInt64) {
		panic("batch cost overflow")
	}
	duration := max(1, int64(math.Ceil(value)))
	if m.observe != nil {
		observation.DurationUS = duration
		m.observe(observation)
	}
	return duration
}

func (m *batchCostLatency) QueueingTime(*sim.Request) int64 { return m.config.EnqueueUS }
func (*batchCostLatency) OutputTokenProcessingTime() int64  { return 0 }
func (*batchCostLatency) PostDecodeFixedOverhead() int64    { return 0 }
