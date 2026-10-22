package kv

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// EngineStepTiming is a prediction from independent profiling. Offsets are
// relative to the start of this engine step, excluding explicit store/load
// submission service (charged separately). OutputReady and GPUReady can differ:
// output copies need not wait for all main-stream postprocessing.
type EngineStepTiming struct {
	PreForwardUS  int64 `json:"pre_forward_us"`
	PostForwardUS int64 `json:"post_forward_us"`
	GPUReadyUS    int64 `json:"gpu_ready_us"`
	OutputReadyUS int64 `json:"output_ready_us"`
	PollUS        int64 `json:"poll_us"`
	TailUS        int64 `json:"tail_us"`
}

type EnginePhaseModel interface {
	PredictEngineStep([]sim.BatchWork) EngineStepTiming
	HostSubmitUS(direction string, blocks int64) int64
}

type phaseJob struct {
	job          *peerJob
	physicalDone bool
}

// PeerEnginePhases owns engine control ordering, never request policy or token
// advancement. Source/target leases and publication remain PeerCache callbacks.
type PeerEnginePhases struct {
	store        *PeerCache
	model        EnginePhaseModel
	observe      func(start, end int64, work []sim.BatchWork)
	staged       []*peerJob
	jobs         map[int64]*phaseJob
	active       bool
	lastGPUReady int64
	flush        map[int64]bool
	afterFlush   func(int64)
}

func (s *PeerCache) ConfigureEnginePhases(model EnginePhaseModel, observe func(int64, int64, []sim.BatchWork)) error {
	f := s.fabric
	if model == nil || f.phases != nil || f.native != nil || f.external != nil || f.Pending() != 0 || len(f.stores) != 1 {
		return fmt.Errorf("engine phases require one idle abstract instance and a cost model")
	}
	m := f.mechanisms
	if m.DirectoryUS != 0 || m.CompletionUS != 0 || m.PolicyUS != 0 || m.ReserveAtDispatch || m.Online != nil || m.CancelQueuedPromotions || (m.TransferPolicy != "" && m.TransferPolicy != "fifo") {
		return fmt.Errorf("engine phases cannot combine with separate control costs, promotion, dispatch reservation or transfer reordering")
	}
	if len(s.access) > 0 && (!m.GroupTransfers || !m.ConcurrentRestores || m.BackgroundStorePool == "") {
		return fmt.Errorf("offload engine phases require grouped concurrent restores and a background store pool")
	}
	f.phases = &PeerEnginePhases{store: s, model: model, observe: observe, jobs: map[int64]*phaseJob{}}
	return nil
}

func (s *PeerCache) ExecutesEmptySteps() bool { return s.fabric.phases != nil }
func (s *PeerCache) HasPendingEngineWork() bool {
	return s.fabric.phases != nil && len(s.fabric.phases.jobs) > 0
}

type peerEnginePhaseEvent struct {
	at       int64
	priority int
	run      func(int64)
}

func (e *peerEnginePhaseEvent) Timestamp() int64         { return e.at }
func (e *peerEnginePhaseEvent) Priority() int            { return e.priority }
func (e *peerEnginePhaseEvent) Execute(_ *sim.Simulator) { e.run(e.at) }

func (p *PeerEnginePhases) schedule(at int64, run func(int64)) {
	p.store.schedule(&peerEnginePhaseEvent{at: at, priority: sim.PriorityStep, run: run})
}
func (p *PeerEnginePhases) emit(at int64, name string, j *peerJob) {
	r := PeerRecord{Time: at, Name: name, Instance: p.store.id}
	if j != nil {
		r = j.record
		r.Time, r.Name = at, name
	}
	p.store.fabric.emit(r)
}
func (p *PeerEnginePhases) stage(now int64, j *peerJob) {
	p.jobs[j.record.Transaction] = &phaseJob{job: j}
	p.staged = append(p.staged, j)
	p.emit(now, "transfer_staged", j)
	if !p.active {
		p.store.wake(now)
	}
}

// submit schedules serialized CPU submission, followed by eligibility on the
// existing directional DMA resource. Each job waits for its stream dependency.
func (p *PeerEnginePhases) submit(now, gpuReady int64, load bool) int64 {
	var selected, remaining []*peerJob
	for _, j := range p.staged {
		if (j.record.Destination == "hbm") == load {
			selected = append(selected, j)
		} else {
			remaining = append(remaining, j)
		}
	}
	p.staged = remaining
	end := now
	for _, j := range selected {
		direction := "d2h"
		if load {
			direction = "h2d"
		}
		cost := p.model.HostSubmitUS(direction, max(1, j.units))
		if cost <= 0 {
			panic("engine phase submission cost must be positive")
		}
		start := end
		end += cost
		p.schedule(start, func(at int64) { p.emit(at, "transfer_submit_begin", j) })
		p.schedule(end, func(at int64) {
			p.emit(at, "transfer_submit_end", j)
			j.notBefore = max(at, gpuReady)
			j.record.Time, j.record.Name = at, "transfer_queued"
			p.store.fabric.emit(j.record)
			p.store.fabric.queue = append(p.store.fabric.queue, j)
			if j.notBefore > at {
				p.schedule(j.notBefore, func(t int64) { p.store.fabric.dispatch(t) })
			}
			p.store.fabric.dispatch(at)
		})
	}
	return end
}

func (p *PeerEnginePhases) physicalComplete(now int64, j *peerJob) {
	state := p.jobs[j.record.Transaction]
	if state == nil || state.physicalDone {
		panic("duplicate or unknown physical completion")
	}
	state.physicalDone = true
	if p.flush[j.record.Transaction] {
		delete(p.flush, j.record.Transaction)
		if len(p.flush) == 0 && p.afterFlush != nil {
			resume := p.afterFlush
			p.afterFlush = nil
			p.schedule(now, resume)
		}
	}
}

func (p *PeerEnginePhases) completed() []*peerJob {
	var jobs []*peerJob
	for _, state := range p.jobs {
		if state.physicalDone {
			jobs = append(jobs, state.job)
		}
	}
	sort.Slice(jobs, func(i, j int) bool { return jobs[i].record.Transaction < jobs[j].record.Transaction })
	return jobs
}

func (p *PeerEnginePhases) begin(now int64, work []sim.BatchWork, done func(int64)) (bool, error) {
	if p.active {
		return false, fmt.Errorf("overlapping engine phase steps")
	}
	if len(work) == 0 && len(p.jobs) == 0 {
		return false, nil
	}
	t := p.model.PredictEngineStep(work)
	if t.PreForwardUS < 0 || t.PostForwardUS < t.PreForwardUS || t.GPUReadyUS < 0 || t.OutputReadyUS < 0 || t.PollUS <= 0 || t.TailUS <= 0 {
		return false, fmt.Errorf("invalid engine phase timing")
	}
	p.active = true
	p.emit(now, "engine_step_begin", nil)
	p.schedule(now+t.PreForwardUS, func(pre int64) {
		p.emit(pre, "engine_pre_forward", nil)
		storeEnd := p.submit(pre, p.lastGPUReady, false)
		p.schedule(storeEnd, func(at int64) {
			resume := func(ready int64) {
				// Explicit host store/flush delays shift subsequent CPU and GPU work.
				delay := ready - pre
				gpuBase := max(now, p.lastGPUReady)
				gpuReady := gpuBase + t.GPUReadyUS + delay
				outputReady := gpuBase + t.OutputReadyUS + delay
				if len(work) == 0 {
					gpuReady, outputReady = p.lastGPUReady, now
				}
				p.lastGPUReady = max(p.lastGPUReady, gpuReady)
				post := now + t.PostForwardUS + delay
				p.schedule(post, func(at int64) {
					p.emit(at, "engine_post_forward", nil)
					loadEnd := p.submit(at, gpuReady, true)
					p.schedule(loadEnd+t.PollUS, func(poll int64) {
						p.emit(poll, "engine_worker_poll", nil)
						completed := p.completed() // only this poll's snapshot can be adopted
						end := max(poll, outputReady) + t.TailUS
						p.schedule(end, func(at int64) {
							for _, j := range completed {
								p.emit(at, "transfer_adopted", j)
								j.done(at)
								delete(p.jobs, j.record.Transaction)
							}
							p.emit(at, "engine_adopt", nil)
							p.store.fabric.wakeStores(at)
							p.active = false
							if p.observe != nil {
								p.observe(now, at, work)
							}
							done(at)
						})
					})
				})
			}
			// Reclaim stores can fence forward progress. Wait for the actual
			// blocking transactions; publication still occurs through poll/adoption.
			p.flush = map[int64]bool{}
			for id, state := range p.jobs {
				j := state.job
				if !state.physicalDone && j.record.Destination != "hbm" && j.blockedRequests != nil && len(j.blockedRequests()) > 0 {
					p.flush[id] = true
				}
			}
			if len(p.flush) > 0 {
				p.emit(at, "engine_flush_wait", nil)
				p.afterFlush = func(t int64) { p.emit(t, "engine_flush_ready", nil); resume(t) }
			} else {
				resume(at)
			}
		})
	})
	return true, nil
}
