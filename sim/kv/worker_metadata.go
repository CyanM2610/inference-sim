package kv

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// EngineWorkerMetadataModel consumes actual worker identity state. The boolean
// slice corresponds to work, not KV prefix length or proposed token grants.
type EngineWorkerMetadataModel interface {
	WorkerMetadataEnabled() bool
	PredictEngineStepWithMetadata(work []sim.BatchWork, fresh []bool) (EngineStepTiming, error)
}

func (p *PeerEnginePhases) predictWorkerStep(work []sim.BatchWork) (EngineStepTiming, []bool, error) {
	model, enabled := p.model.(EngineWorkerMetadataModel)
	if !enabled || !model.WorkerMetadataEnabled() {
		return p.model.PredictEngineStep(work), nil, nil
	}
	fresh := make([]bool, len(work))
	seen := map[string]bool{}
	for i, w := range work {
		if w.Request == nil || w.Request.ID == "" || seen[w.Request.ID] || w.NewTokens <= 0 {
			return EngineStepTiming{}, nil, fmt.Errorf("worker metadata requires unique actual compute requests")
		}
		seen[w.Request.ID] = true
		previous := p.workerResidents[w.Request.ID]
		if previous != nil && previous != w.Request {
			return EngineStepTiming{}, nil, fmt.Errorf("worker request identity reused without release: %s", w.Request.ID)
		}
		fresh[i] = previous == nil
	}
	timing, err := model.PredictEngineStepWithMetadata(work, append([]bool(nil), fresh...))
	return timing, fresh, err
}

func (p *PeerEnginePhases) commitWorkerStep(now int64, work []sim.BatchWork, fresh []bool, timing EngineStepTiming) {
	if p.workerResidents == nil || len(work) == 0 {
		return
	}
	var added []string
	var reusedPrefix int64
	for i, w := range work {
		if fresh[i] {
			added = append(added, w.Request.ID)
			if w.PrefixTokens > 0 {
				reusedPrefix++
			}
		}
		p.workerResidents[w.Request.ID] = w.Request
	}
	p.store.fabric.emit(PeerRecord{Time: now, Name: "engine_worker_metadata", Instance: p.store.id,
		Requests: added, Counters: map[string]int64{"requests": int64(len(work)), "new_requests": int64(len(added)),
			"resident_requests": int64(len(work) - len(added)), "new_with_prefix": reusedPrefix,
			"predicted_pre_forward_us": timing.PreForwardUS, "predicted_post_forward_us": timing.PostForwardUS,
			"predicted_gpu_ready_us": timing.GPUReadyUS, "predicted_output_ready_us": timing.OutputReadyUS,
			"predicted_poll_us": timing.PollUS, "predicted_tail_us": timing.TailUS}})
}

func (s *PeerCache) releaseWorkerRequest(req *sim.Request) {
	p := s.fabric.phases
	if p == nil || p.workerResidents == nil || p.workerResidents[req.ID] == nil {
		return
	}
	if p.workerResidents[req.ID] != req {
		panic("worker release identity differs")
	}
	delete(p.workerResidents, req.ID)
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "worker_request_released", Instance: s.id, Request: req.ID})
}
