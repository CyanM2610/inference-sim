package kv

import (
	"fmt"
	"slices"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

const StoreSourceReuseContract = "all_stores_fenced_v1"

// sourceOverwrite survives transfer grouping before a transaction ID exists.
// It records an actual allocation, never a predicted future overwrite.
type sourceOverwrite struct {
	at      int64
	request string
	block   int64
}

func (s *PeerCache) storeSourceReuseEnabled() bool {
	return s.fabric.phases != nil && s.fabric.phases.reuseSources
}

// EnableStoreSourceReuse permits allocator reuse for all STORE sources,
// including on-reclaim victims. BeginBatch must fence both compute and LOAD
// writes until the old source's physical copy finishes; publication still
// waits for worker poll/adoption. It cannot be enabled after execution starts.
func (s *PeerCache) EnableStoreSourceReuse() error {
	p := s.fabric.phases
	if p == nil || p.active || p.started || s.fabric.Pending() != 0 || s.fabric.groupDepth != 0 {
		return fmt.Errorf("store source reuse requires unused engine phases")
	}
	p.reuseSources = true
	p.reuseFences = map[int64]bool{}
	return nil
}

func (p *PeerEnginePhases) recordSourceFence(j *peerJob, d sourceOverwrite) {
	p.reuseFences[j.record.Transaction] = true
	r := j.record
	r.Time, r.Name, r.Request, r.HBMBlocks = d.at, "hbm_reuse_dependency", d.request, []int64{d.block}
	p.store.fabric.emit(r)
}

// allocated is called only for newly assigned physical slots, not read-only
// prefix hits. Reclaim can create a STORE inside the same transfer group as
// this allocation, so registration must include jobs not yet assigned an ID.
func (p *PeerEnginePhases) allocated(now int64, request string, blocks []int64) {
	record := func(j *peerJob, staged bool) {
		for _, block := range blocks {
			if !slices.Contains(j.record.HBMBlocks, block) {
				continue
			}
			if slices.ContainsFunc(j.overwrites, func(d sourceOverwrite) bool { return d.request == request && d.block == block }) {
				continue
			}
			d := sourceOverwrite{at: now, request: request, block: block}
			j.overwrites = append(j.overwrites, d)
			if staged {
				p.recordSourceFence(j, d)
			}
		}
	}
	ids := make([]int64, 0, len(p.jobs))
	for id := range p.jobs {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		if state := p.jobs[id]; !state.physicalDone {
			record(state.job, true)
		}
	}
	for _, j := range p.store.fabric.groupBuffer {
		record(j, false)
	}
}

func (s *PeerCache) ExecutionDependencies(request string) []sim.KVExecutionDependency {
	p := s.fabric.phases
	if p == nil || !p.reuseSources {
		return nil
	}
	var out []sim.KVExecutionDependency
	for id, state := range p.jobs {
		if state.physicalDone {
			continue
		}
		var blocks []int64
		for _, d := range state.job.overwrites {
			if d.request == request && !slices.Contains(blocks, d.block) {
				blocks = append(blocks, d.block)
			}
		}
		if len(blocks) != 0 {
			slices.Sort(blocks)
			out = append(out, sim.KVExecutionDependency{Kind: "store_before_overwrite", Transaction: id, HBMBlocks: blocks})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Transaction < out[j].Transaction })
	return out
}
