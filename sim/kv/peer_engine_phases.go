package kv

import (
	"fmt"
	"math"
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

// EngineLoadPlanningModel optionally accounts for creating new load metadata
// above the base engine phase. Jobs already submitted or merely polled do not
// repeat this work. Copy submission and device service remain separate.
type EngineLoadPlanningModel interface {
	PredictLoadPlanning(loads, blocks int64) (int64, error)
}

type phaseJob struct {
	job          *peerJob
	physicalDone bool
	submitted    bool
	predecessor  *phaseJob
}

// PeerEnginePhases owns engine control ordering, never request policy or token
// advancement. Source/target leases and publication remain PeerCache callbacks.
type PeerEnginePhases struct {
	submissionTails   map[string]*phaseJob
	started           bool
	submissionPolicy  TransferSubmissionPolicy
	submissionCost    *TransferSubmissionCost
	observeSubmission func(TransferSubmissionRecord)
	workerResidents   map[string]*sim.Request
	completionBarrier sim.HostCompletionBarrier
	store             *PeerCache
	model             EnginePhaseModel
	observe           func(start, end int64, work []sim.BatchWork)
	staged            []*peerJob
	jobs              map[int64]*phaseJob
	active            bool
	lastGPUReady      int64
	flush             map[int64]bool
	afterFlush        func(int64)
	reuseSources      bool
	reuseFences       map[int64]bool
	preemptionFences  map[int64]bool
}

// Pinned vLLM flushes every existing store owned by a preempted request,
// including stores whose HBM source is merely shared, not overwritten.
func (s *PeerCache) PreparePrefillPreemption(req *sim.Request) {
	p := s.fabric.phases
	if p == nil {
		return
	}
	if p.preemptionFences == nil {
		p.preemptionFences = map[int64]bool{}
	}
	ids := make([]int64, 0)
	for id, state := range p.jobs {
		if state.job.record.Request == req.ID && state.job.record.Destination != "hbm" {
			ids = append(ids, id)
		}
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		p.preemptionFences[id] = true
		p.emit(s.clock, "preemption_store_dependency", p.jobs[id].job)
	}
}

// EnableStoreSourceReuse separates allocator ownership from an outstanding
// background copy's content lease. Only the phase backend can fence overwrite.
func (s *PeerCache) EnableStoreSourceReuse() error {
	p := s.fabric.phases
	if p == nil || p.active || s.fabric.Pending() != 0 {
		return fmt.Errorf("store source reuse requires idle engine phases")
	}
	p.reuseSources = true
	p.reuseFences = map[int64]bool{}
	return nil
}

// allocated records a dependency at actual physical reuse, after successful
// compute allocation or load reservation. A prefix hit only reads the old
// content and does not fence its background STORE. No future times are read.
func (p *PeerEnginePhases) allocated(now int64, request string, blocks []int64) {
	ids := make([]int64, 0, len(p.jobs))
	for id := range p.jobs {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		state := p.jobs[id]
		if state.physicalDone {
			continue
		}
		for _, block := range blocks {
			for _, source := range state.job.record.HBMBlocks {
				if block == source {
					p.reuseFences[id] = true
					r := state.job.record
					r.Time, r.Name, r.Request, r.HBMBlocks = now, "hbm_reuse_dependency", request, []int64{block}
					p.store.fabric.emit(r)
				}
			}
		}
	}
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
	if m, ok := model.(EngineWorkerMetadataModel); ok && m.WorkerMetadataEnabled() {
		f.phases.workerResidents = map[string]*sim.Request{}
	}
	return nil
}

func (s *PeerCache) ExecutesEmptySteps() bool { return s.fabric.phases != nil }

func (s *PeerCache) BindHostCompletionBarrier(barrier sim.HostCompletionBarrier) error {
	p := s.fabric.phases
	if p == nil || p.active || p.completionBarrier != nil || barrier == nil {
		return fmt.Errorf("host completion barrier requires idle engine phases")
	}
	p.completionBarrier = barrier
	return nil
}
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
	p.started = true
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
	direction := "d2h"
	if load {
		direction = "h2d"
	}
	selected, record, err := p.orderSubmissions(now, direction, selected)
	if err != nil {
		panic(err)
	}
	end := now
	if record != nil {
		end = record.ServiceEndUS
	}
	// Validate all service before consuming staged work; invalid callbacks or
	// costs cannot leave a partially scheduled phase.
	costs := make([]int64, len(selected))
	for i, j := range selected {
		cost := p.model.HostSubmitUS(direction, max(1, j.units))
		if cost <= 0 || end < 0 || cost > math.MaxInt64-end {
			panic("engine phase submission cost must be positive and fit the clock")
		}
		costs[i], end = cost, end+cost
	}
	if record != nil {
		record.SubmitEndUS = end
	}
	p.staged = remaining
	end = now
	if record != nil {
		end = record.ServiceEndUS
	}
	for i, j := range selected {
		start := end
		end += costs[i]
		p.schedule(start, func(at int64) { p.emit(at, "transfer_submit_begin", j) })
		p.schedule(end, func(at int64) {
			p.emit(at, "transfer_submit_end", j)
			p.linkSubmission(at, direction, j)
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
	if record != nil && p.observeSubmission != nil {
		p.observeSubmission(*record)
	}
	return end
}

func (p *PeerEnginePhases) physicalComplete(now int64, j *peerJob) {
	state := p.jobs[j.record.Transaction]
	if state == nil || state.physicalDone {
		panic("duplicate or unknown physical completion")
	}
	state.physicalDone = true
	state.predecessor = nil
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
	t, fresh, err := p.predictWorkerStep(work)
	if err != nil {
		return false, err
	}
	if t.PreForwardUS < 0 || t.PostForwardUS < t.PreForwardUS || t.GPUReadyUS < 0 || t.OutputReadyUS < 0 || t.PollUS <= 0 || t.TailUS <= 0 {
		return false, fmt.Errorf("invalid engine phase timing")
	}
	var loads, blocks, planning int64
	for _, j := range p.staged {
		if j.record.Destination == "hbm" {
			loads++
			blocks += max(1, j.units)
		}
	}
	if model, ok := p.model.(EngineLoadPlanningModel); ok && loads > 0 {
		var err error
		planning, err = model.PredictLoadPlanning(loads, blocks)
		if err != nil || planning < 0 {
			return false, fmt.Errorf("invalid engine load planning: cost=%d error=%v", planning, err)
		}
	}
	base := now + planning
	if base < now {
		return false, fmt.Errorf("engine load planning clock overflow")
	}
	p.started, p.active = true, true
	p.commitWorkerStep(now, work, fresh, t)
	p.emit(now, "engine_step_begin", nil)
	if planning > 0 {
		p.store.fabric.emit(PeerRecord{Time: now, Name: "engine_load_planning", Instance: p.store.id, Duration: planning, Counters: map[string]int64{"loads": loads, "blocks": blocks}})
	}
	p.schedule(base+t.PreForwardUS, func(pre int64) {
		p.emit(pre, "engine_pre_forward", nil)
		storeEnd := p.submit(pre, p.lastGPUReady, false)
		p.schedule(storeEnd, func(at int64) {
			resume := func(ready int64) {
				if p.reuseSources {
					p.emit(ready, "engine_forward_ready", nil)
				}
				// Explicit host store/flush delays shift subsequent CPU and GPU work.
				delay := ready - pre
				gpuBase := max(base, p.lastGPUReady)
				gpuReady := gpuBase + t.GPUReadyUS + delay
				outputReady := gpuBase + t.OutputReadyUS + delay
				if len(work) == 0 {
					gpuReady, outputReady = p.lastGPUReady, now
				}
				p.lastGPUReady = max(p.lastGPUReady, gpuReady)
				post := base + t.PostForwardUS + delay
				p.schedule(post, func(at int64) {
					p.emit(at, "engine_post_forward", nil)
					p.store.afterHotPrefixPlacement(at, work, func(at int64) {
						loadEnd := p.submit(at, gpuReady, true)
						p.schedule(loadEnd+t.PollUS, func(poll int64) {
							p.emit(poll, "engine_worker_poll", nil)
							completed := p.completed() // only this poll's snapshot can be adopted
							end := max(poll, outputReady) + t.TailUS
							p.schedule(end, func(at int64) {
								publish := func(at int64) {
									for _, j := range completed {
										p.emit(at, "transfer_adopted", j)
										j.done(at)
										delete(p.jobs, j.record.Transaction)
										p.store.fabric.notifyDecisionTransfer(at, j)
									}
									p.emit(at, "engine_adopt", nil)
									p.store.fabric.wakeStores(at)
									p.active = false
									if p.observe != nil {
										p.observe(now, at, work)
									}
									done(at)
								}
								if p.completionBarrier != nil {
									p.completionBarrier(at, work, publish)
								} else {
									publish(at)
								}
							})
						})
					})
				})
			}
			// Reclaim stores can fence forward progress. Wait for the actual
			// blocking transactions; publication still occurs through poll/adoption.
			p.flush = map[int64]bool{}
			for id, state := range p.jobs {
				j := state.job
				blocked := j.blockedRequests != nil && len(j.blockedRequests()) > 0
				if !state.physicalDone && j.record.Destination != "hbm" && (blocked || p.reuseFences[id] || p.preemptionFences[id]) {
					p.flush[id] = true
				}
			}
			p.preemptionFences = nil
			if p.reuseSources {
				p.reuseFences = map[int64]bool{}
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
