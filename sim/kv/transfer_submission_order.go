package kv

import (
	"fmt"
	"math"
)

// EnableDirectionalTransferOrder models the pinned native handler's event chain:
// a submitted copy waits for the preceding copy of the same direction to finish
// physically. Adoption is a separate later operation. Resource paths still decide
// whether opposite directions can overlap. This must be configured before work.
func (s *PeerCache) EnableDirectionalTransferOrder() error {
	p := s.fabric.phases
	if p == nil || p.store != s || p.started || p.active || s.fabric.Pending() != 0 || p.submissionTails != nil {
		return fmt.Errorf("directional transfer order requires unused configured engine phases")
	}
	p.submissionTails = map[string]*phaseJob{}
	return nil
}

// linkSubmission runs at actual host submission, not job creation or policy
// evaluation. A reference remains valid after the predecessor is adopted and
// removed from jobs. Completed nodes drop their backward reference to bound
// retention; only the two latest tails and currently pending jobs remain live.
func (p *PeerEnginePhases) linkSubmission(now int64, direction string, j *peerJob) {
	if p.submissionTails == nil {
		return
	}
	state := p.jobs[j.record.Transaction]
	if state == nil || state.submitted {
		panic("duplicate or unknown phase submission")
	}
	state.submitted = true
	state.predecessor = p.submissionTails[direction]
	p.submissionTails[direction] = state
	if state.predecessor != nil {
		r := j.record
		r.Time, r.Name, r.Value = now, "transfer_submission_dependency", state.predecessor.job.record.Transaction
		p.store.fabric.emit(r)
	}
}

func (p *PeerEnginePhases) submissionReady(j *peerJob) bool {
	if p.submissionTails == nil {
		return true
	}
	state := p.jobs[j.record.Transaction]
	if state == nil || !state.submitted {
		panic("unsubmitted phase job reached fabric")
	}
	return state.predecessor == nil || state.predecessor.physicalDone
}

// estimateQueueUS projects only already submitted work using the configured
// service and GPU-ready predictions. It observes the dependency graph without
// advancing it. Unresolved HBM/host work and future submissions remain excluded.
func (p *PeerEnginePhases) estimateQueueUS(now int64, path []string) int64 {
	f := p.store.fabric
	until := map[string]int64{}
	for id, at := range f.busyUntil {
		until[id] = max(now, at)
	}
	queue := append([]*peerJob(nil), f.queue...)
	queued := map[*phaseJob]bool{}
	for _, j := range queue {
		queued[p.jobs[j.record.Transaction]] = true
	}
	finish := map[*phaseJob]int64{}
	for _, state := range p.jobs {
		if state.physicalDone {
			finish[state] = now
			continue
		}
		if state.submitted && !queued[state] {
			// Active copies have a modeled resource reservation. No native
			// completion timestamp or output history is consulted here.
			end := now
			for _, id := range state.job.path {
				end = max(end, until[id])
			}
			finish[state] = end
		}
	}
	for len(queue) > 0 {
		best := -1
		first := int64(math.MaxInt64)
		for i, j := range queue {
			state := p.jobs[j.record.Transaction]
			start := max(now, j.notBefore)
			if previous := state.predecessor; previous != nil && !previous.physicalDone {
				end, known := finish[previous]
				if !known {
					continue
				}
				start = max(start, end)
			}
			for _, id := range j.path {
				start = max(start, until[id])
			}
			if best < 0 || start < first || start == first && f.jobLess(j, queue[best]) {
				best, first = i, start
			}
		}
		if best < 0 {
			panic("submitted transfer graph has an unresolved predecessor")
		}
		j := queue[best]
		duration := f.durationBytes(j.path, max(1, j.units)*f.bytes)
		end := int64(math.MaxInt64)
		if duration <= math.MaxInt64-first {
			end = first + duration
		}
		finish[p.jobs[j.record.Transaction]] = end
		for _, id := range j.path {
			until[id] = end
		}
		queue = append(queue[:best], queue[best+1:]...)
	}
	start := now
	for _, id := range path {
		start = max(start, until[id])
	}
	return start - now
}
