package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

const RestoreEstimateCoverage = "profile_prediction_new_policy_unvalidated_no_prefill_interference_or_reclaim_model"

type DecisionEstimateConfig struct {
	Spill                  bool  `json:"spill,omitempty"`
	MaxDecodeContextTokens int64 `json:"max_decode_context_tokens"`
}

type restoreEstimator struct {
	spill      bool
	prefillCap int64
	phase      EnginePhaseCostConfig
	resources  map[string]kv.PeerResource
	access     kv.PeerAccess
	window     int64
	maxContext int64
}

// NewRestoreDecisionEstimator exposes the same read-only predictor used by Run
// to offline adapters and contract checks. It does not construct a simulator.
func NewRestoreDecisionEstimator(c Config) (sim.DecisionEstimator, error) {
	return newRestoreEstimator(c)
}

func newRestoreEstimator(c Config) (*restoreEstimator, error) {
	if c.DecisionEstimates == nil || c.DecisionEstimates.MaxDecodeContextTokens <= 0 || !c.RestoreControl || c.EnginePhases == nil || c.Mechanisms == nil || len(c.Instances) != 1 || len(c.Instances[0].Access) != 1 {
		return nil, fmt.Errorf("decision estimates require restore control, engine phases, one CPU access path and positive max decode context")
	}
	if c.EnginePhases.WorkerMetadata != nil {
		if err := c.EnginePhases.WorkerMetadata.Validate(); err != nil {
			return nil, err
		}
	}
	m := &restoreEstimator{phase: *c.EnginePhases, access: c.Instances[0].Access[0], window: int64(max(1, c.Mechanisms.RestoreWindow)), maxContext: c.DecisionEstimates.MaxDecodeContextTokens, resources: map[string]kv.PeerResource{}}
	if c.DecisionPolicy != nil {
		m.prefillCap = c.DecisionPolicy.PrefillTokenCap
	}
	if c.DecisionEstimates.Spill {
		if c.DecisionPolicy == nil || !c.DecisionPolicy.RequestSpill || !c.Mechanisms.GroupTransfers {
			return nil, fmt.Errorf("spill estimates require request_spill and grouped transfers")
		}
		m.spill = true
	}
	for _, r := range c.Resources {
		m.resources[r.ID] = r
	}
	if len(m.access.ReadPath) == 0 {
		return nil, fmt.Errorf("empty restore prediction path")
	}
	for _, path := range [][]string{m.access.ReadPath, m.access.WritePath} {
		for _, id := range path {
			if _, ok := m.resources[id]; !ok {
				return nil, fmt.Errorf("unknown prediction resource %q", id)
			}
		}
	}
	return m, nil
}

func estimateAdd(a, b int64) int64 {
	if a < 0 || b < 0 || a >= math.MaxInt64/16-b {
		panic("restore prediction overflow")
	}
	return a + b
}

func (m *restoreEstimator) pathUS(path []string, bytes int64) int64 {
	var fixed int64
	var transfer float64
	for _, id := range path {
		r := m.resources[id]
		fixed = estimateAdd(fixed, r.LatencyUS)
		transfer = max(transfer, float64(bytes)/r.BytesPerUS)
	}
	if math.IsInf(transfer, 0) || math.IsNaN(transfer) || transfer >= float64(math.MaxInt64/16) {
		panic("restore path estimate overflow")
	}
	return max(1, estimateAdd(fixed, int64(math.Ceil(transfer))))
}

// Remaining service is not observable. Charge the full service of each public
// pending job as a prediction; no physicalDone, busyUntil or future event reads.
func (m *restoreEstimator) backlogUS(v sim.DecisionView) (int64, error) {
	return m.backlogForPathUS(v, m.access.ReadPath)
}

func (m *restoreEstimator) backlogForPathUS(v sim.DecisionView, targetPath []string) (int64, error) {
	var backlog int64
	for _, t := range v.KV.PendingTransfers {
		path, direction := m.access.WritePath, "d2h"
		if t.Destination == "hbm" {
			path, direction = m.access.ReadPath, "h2d"
		}
		if (t.Destination == "hbm" && t.Source != m.access.Pool) || (t.Destination != "hbm" && t.Destination != m.access.Pool) || t.Bytes <= 0 {
			return 0, fmt.Errorf("unknown pending transfer path in prediction")
		}
		blocks := (t.Bytes-1)/m.phase.BlockBytes + 1
		backlog = estimateAdd(backlog, m.phase.HostSubmitUS(direction, blocks))
		shared := false
		for _, a := range path {
			for _, b := range targetPath {
				shared = shared || a == b
			}
		}
		if shared {
			backlog = estimateAdd(backlog, m.pathUS(path, t.Bytes))
		}
	}
	return backlog, nil
}

func (m *restoreEstimator) loadUS(blocks, backlog int64) (int64, error) {
	if blocks == 0 {
		return 0, nil
	}
	total := backlog
	idle := m.phase.Idle
	for blocks > 0 {
		n := min(blocks, m.window)
		if n > math.MaxInt64/m.phase.BlockBytes {
			panic("restore bytes overflow")
		}
		// One submit and an additional idle polling cycle cover phase adoption
		// lag for this isolated estimate. Concurrent forward gates are not known.
		service := estimateAdd(idle.PostForwardUS, m.phase.HostSubmitUS("h2d", n))
		planning, err := m.phase.PredictLoadPlanning(1, n)
		if err != nil {
			return 0, err
		}
		service = estimateAdd(service, planning)
		service = estimateAdd(service, m.pathUS(m.access.ReadPath, n*m.phase.BlockBytes))
		service = estimateAdd(service, estimateAdd(idle.PostForwardUS, estimateAdd(idle.PollUS, idle.TailUS)))
		total = estimateAdd(total, service)
		blocks -= n
	}
	return total, nil
}

func (m *restoreEstimator) computeUS(v sim.DecisionView, r sim.DecisionRequest, prefix int64, contexts []int64) (int64, error) {
	newWorker := false
	if m.phase.WorkerMetadataEnabled() {
		if !v.KV.WorkerStateKnown {
			return 0, fmt.Errorf("worker metadata prediction requires known worker state")
		}
		found := false
		for _, q := range v.KV.Requests {
			if q.ID == r.ID {
				newWorker, found = !q.WorkerResident, true
				break
			}
		}
		if !found {
			return 0, fmt.Errorf("missing worker state for %q", r.ID)
		}
	}
	budget := v.MaxBatchTokens - int64(len(contexts))
	if budget <= 0 {
		return 0, fmt.Errorf("decode-load estimate leaves no prefill token capacity")
	}
	chunk := budget
	if v.PrefillChunk > 0 {
		chunk = min(chunk, v.PrefillChunk)
	}
	if m.prefillCap > 0 {
		chunk = min(chunk, m.prefillCap)
	}
	var now, gpuReady, steps int64
	for prefix < r.InputTokens {
		query := min(chunk, r.InputTokens-prefix)
		work := []sim.BatchWork{{PrefixTokens: prefix, NewTokens: query}}
		for _, context := range contexts {
			work = append(work, sim.BatchWork{PrefixTokens: estimateAdd(context, steps), NewTokens: 1})
		}
		var t kv.EngineStepTiming
		if m.phase.WorkerMetadataEnabled() {
			fresh := make([]bool, len(work))
			fresh[0] = steps == 0 && newWorker
			var err error
			t, err = m.phase.PredictEngineStepWithMetadata(work, fresh)
			if err != nil {
				return 0, err
			}
		} else {
			t = m.phase.PredictEngineStep(work)
		}
		gpuBase := max(now, gpuReady)
		gpuReady = estimateAdd(gpuBase, t.GPUReadyUS)
		output := estimateAdd(gpuBase, t.OutputReadyUS)
		poll := estimateAdd(now, estimateAdd(t.PostForwardUS, t.PollUS))
		now = estimateAdd(max(output, poll), t.TailUS)
		prefix += query
		steps++
	}
	return now, nil
}

func (m *restoreEstimator) Estimate(v sim.DecisionView) (sim.DecisionEstimates, error) {
	e := sim.DecisionEstimates{Provenance: m.phase.Provenance + "; configured transfer resources; full public backlog; configured maximum decode context", Coverage: RestoreEstimateCoverage}
	if !v.KV.PrefixStateKnown || !v.KV.TransfersKnown || v.BlockTokens <= 0 || v.MaxSequences <= 0 {
		return e, fmt.Errorf("restore estimates require known prefix and pending-transfer state")
	}
	if m.spill {
		if err := m.addSpillEstimates(v, &e); err != nil {
			return e, err
		}
	}
	backlog, err := m.backlogUS(v)
	if err != nil {
		return e, err
	}
	states := map[string]sim.DecisionKVRequest{}
	for _, q := range v.KV.Requests {
		states[q.ID] = q
	}
	var actual []int64
	for _, r := range v.Running {
		if r.ComputedTokens >= r.InputTokens {
			if m.phase.WorkerMetadataEnabled() && (!v.KV.WorkerStateKnown || !states[r.ID].WorkerResident) {
				return e, fmt.Errorf("decode companion requires known resident worker state")
			}
			actual = append(actual, r.ComputedTokens)
		}
	}
	maximum := make([]int64, max(0, v.MaxSequences-1))
	for i := range maximum {
		maximum[i] = m.maxContext
	}
	for _, r := range v.Waiting {
		if r.ComputedTokens >= max(r.InputTokens, r.RecomputeUntilTokens) {
			continue
		}
		row := sim.DecisionRequestEstimate{Request: r.ID}
		if r.RecomputeUntilTokens > r.InputTokens {
			row.Unavailable = "decode_recompute_not_profiled"
			e.Requests = append(e.Requests, row)
			continue
		}
		q, ok := states[r.ID]
		if !ok {
			return e, fmt.Errorf("missing prefix state for %q", r.ID)
		}
		if q.TransferPending {
			row.Unavailable = "restore_in_flight"
			e.Requests = append(e.Requests, row)
			continue
		}
		// The candidate may wait for a decode slot; do not pretend to know
		// which running decode will finish. Scheduling delay is out of scope.
		if int64(len(actual)) >= v.MaxSequences {
			row.Unavailable = "no_free_sequence_slot"
			e.Requests = append(e.Requests, row)
			continue
		}
		for end := q.LocalPrefixBlocks; end <= q.RecoverablePrefixBlocks; end++ {
			prefix := q.LocalPrefixTokens
			if end > q.LocalPrefixBlocks {
				prefix = min(end*v.BlockTokens, r.InputTokens-1)
			}
			cost, err := m.computeUS(v, r, prefix, actual)
			if err != nil {
				return e, err
			}
			maxCost, err := m.computeUS(v, r, prefix, maximum)
			if err != nil {
				return e, err
			}
			load, err := m.loadUS(end-q.LocalPrefixBlocks, backlog)
			if err != nil {
				return e, err
			}
			row.Choices = append(row.Choices, sim.DecisionRestoreEstimate{MaxPrefixBlocks: end, LoadUS: load, ComputeUS: cost, MaxLoadComputeUS: max(cost, maxCost)})
		}
		e.Requests = append(e.Requests, row)
	}
	return e, nil
}
