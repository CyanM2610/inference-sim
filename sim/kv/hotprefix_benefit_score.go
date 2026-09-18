package kv

import "slices"

type HotPrefixBenefitEstimate struct {
	SnapshotReadQueueUS  int64               `json:"snapshot_read_queue_us"`
	SnapshotWriteQueueUS int64               `json:"snapshot_write_queue_us"`
	SnapshotGPUWaitUS    int64               `json:"snapshot_gpu_wait_us"`
	KnownStagedJobs      int64               `json:"known_staged_jobs"`
	KnownStagedHostUS    int64               `json:"known_staged_host_us"`
	HostUsedBlocks       int64               `json:"host_used_blocks"`
	Forecast             HotPrefixForecast   `json:"forecast"`
	Keep                 PlacementAccessCost `json:"keep"`
	Evict                PlacementAccessCost `json:"evict"`
	FreeBytes            int64               `json:"free_bytes"`
	NewStoreBlocks       int64               `json:"new_store_blocks"`
	StoreAccepted        []string            `json:"store_accepted,omitempty"`
	StoreRejected        []string            `json:"store_rejected,omitempty"`
	HostVictims          []string            `json:"host_victims,omitempty"`
	CurrentStoreUS       int64               `json:"current_incremental_store_us"`
	NextMarginalUS       int64               `json:"next_marginal_us"`
	ValueUSPerByte       float64             `json:"value_us_per_byte"`
	Coverage             string              `json:"coverage"`
}

func (p *HotPrefixPolicy) prefixPath(tail string) []string {
	var path []string
	for tail != "" {
		n := p.nodes[tail]
		if n == nil {
			panic("benefit path lost prefix ancestry")
		}
		path = append(path, tail)
		tail = n.parent
	}
	slices.Reverse(path)
	return path
}

func (p *HotPrefixPolicy) assessBenefit(g hotPrefixLogicalSegment, state *PlacementCostSnapshot) *HotPrefixBenefitEstimate {
	if len(g.members) == 0 || len(g.members) != len(g.hashes) {
		panic("invalid benefit physical group")
	}
	e := &HotPrefixBenefitEstimate{Forecast: p.nodes[g.hashes[0]].references.predict(state.NowUS, *p.config.Benefit),
		SnapshotReadQueueUS: state.ReadQueueUS, SnapshotWriteQueueUS: state.WriteQueueUS, SnapshotGPUWaitUS: state.GPUWaitUS,
		KnownStagedJobs: state.KnownStagedJobs, KnownStagedHostUS: state.KnownStagedHostUS, HostUsedBlocks: int64(len(state.Host)),
		FreeBytes: int64(len(g.members)) * state.model.blockBytes,
		Coverage:  "conditional_single_reader_one_new_query; exact_physical_group_bytes; current_HP_admission; existing_phase_curves_and_current_queue_projection; pending_store_envelope; unknown_future_arrivals_cobatches_capacity_reclamation_not_predicted; not_native_validation"}
	path := p.prefixPath(g.hashes[0])
	e.Keep = state.access(path, e.Forecast.NextOffsetUS, p.blockTokens)
	dropped := state.clone()
	// Pure simulation of current HP admission/replacement. New reservations
	// are not eligible victims, matching atomic runtime group application.
	for i, a := range g.members {
		hash := g.hashes[i]
		if dropped.HBM[hash] <= 0 {
			panic("benefit counts a nonresident physical block")
		}
		dropped.HBM[hash]--
		if a.Pool == "" {
			continue
		}
		if _, exists := dropped.Host[hash]; exists {
			continue
		}
		n := p.nodes[hash]
		if n.frequency == 0 || n.frequency < p.config.AdmissionThreshold {
			e.StoreRejected = append(e.StoreRejected, hash)
			continue
		}
		if int64(len(dropped.Host)) >= dropped.HostCapacity {
			victim := ""
			for key, copy := range dropped.Host {
				if !copy.Ready || copy.Pinned {
					continue
				}
				if victim == "" || p.hotness(key) < p.hotness(victim) || p.hotness(key) == p.hotness(victim) && key < victim {
					victim = key
				}
			}
			if victim == "" || p.hotness(hash) <= p.hotness(victim) {
				e.StoreRejected = append(e.StoreRejected, hash)
				continue
			}
			delete(dropped.Host, victim)
			e.HostVictims = append(e.HostVictims, victim)
		}
		dropped.Host[hash] = PlacementHostCopy{Pinned: true}
		e.StoreAccepted = append(e.StoreAccepted, hash)
		e.NewStoreBlocks++
	}
	if e.NewStoreBlocks > 0 {
		total := state.BufferedStoreBlocks + e.NewStoreBlocks
		e.CurrentStoreUS = state.model.storeService(total) - state.model.storeService(state.BufferedStoreBlocks)
		ready := state.storeReadyUS(total)
		for _, hash := range e.StoreAccepted {
			dropped.Host[hash] = PlacementHostCopy{Pinned: true, ReadyInUS: ready}
		}
	}
	e.Evict = dropped.access(path, e.Forecast.NextOffsetUS, p.blockTokens)
	e.NextMarginalUS = max(0, e.Evict.TotalUS-e.Keep.TotalUS)
	e.ValueUSPerByte = (float64(e.CurrentStoreUS) + e.Forecast.NextProbability*float64(e.NextMarginalUS)) / float64(e.FreeBytes)
	return e
}
