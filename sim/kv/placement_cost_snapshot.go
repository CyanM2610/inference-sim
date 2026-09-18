package kv

import "fmt"

type PlacementHostCopy struct {
	Ready     bool
	Pinned    bool
	ReadyInUS int64 // current-profile projection, never a future trace lookup
}

// PlacementCostSnapshot contains current visible copy/queue state. The phase
// predictor has no mutable cache or request references. Policy edits to this
// detached snapshot cannot reserve capacity or change actual transfers.
type PlacementCostSnapshot struct {
	NowUS                                int64
	HBM                                  map[string]int
	Host                                 map[string]PlacementHostCopy
	HostCapacity                         int64
	BufferedStoreBlocks                  int64
	ReadQueueUS, WriteQueueUS, GPUWaitUS int64
	RestoreWindow                        int64
	KnownStagedJobs                      int64
	KnownStagedHostUS                    int64
	model                                *PlacementPhaseCosts
}

func (c *PlacementCostSnapshot) clone() *PlacementCostSnapshot {
	if c == nil {
		return nil
	}
	x := *c
	x.HBM = make(map[string]int, len(c.HBM))
	x.Host = make(map[string]PlacementHostCopy, len(c.Host))
	for k, v := range c.HBM {
		x.HBM[k] = v
	}
	for k, v := range c.Host {
		x.Host[k] = v
	}
	return &x
}

func (s *PeerCache) ConfigureBenefitCosts(chunk, decisionUS int64) error {
	if s.hotprefix == nil || s.hotprefix.policy.config.Benefit == nil || s.fabric.phases == nil || !s.restoreControl || s.fabric.Pending() != 0 {
		return fmt.Errorf("benefit costs require an unused HotPrefix forecast and native restore engine")
	}
	var read, write []PeerResource
	for _, id := range s.access[0].ReadPath {
		read = append(read, s.fabric.resources[id])
	}
	for _, id := range s.access[0].WritePath {
		write = append(write, s.fabric.resources[id])
	}
	m, err := newPlacementPhaseCosts(s.fabric.phases.model, chunk, decisionUS, s.fabric.bytes, read, write)
	if err != nil {
		return err
	}
	s.hotprefix.benefitCosts = m
	return nil
}

func (s *PeerCache) benefitCostSnapshot(request string) *PlacementCostSnapshot {
	h := s.hotprefix
	if h == nil || h.policy.config.Benefit == nil {
		return nil
	}
	if h.benefitCosts == nil {
		panic("benefit forecast lacks configured cost model")
	}
	started := h.policy.cpuStart()
	defer h.policy.cpuEnd("benefit_snapshot", started)
	m := h.benefitCosts
	p := s.fabric.phases
	pool := s.fabric.pools[h.pool]
	c := &PlacementCostSnapshot{NowUS: s.clock, HBM: map[string]int{}, Host: map[string]PlacementHostCopy{}, HostCapacity: pool.config.CapacityBlocks,
		RestoreWindow: int64(max(1, s.fabric.mechanisms.RestoreWindow)), model: m,
		ReadQueueUS: s.fabric.EstimateQueueUS(s.clock, s.access[0].ReadPath), WriteQueueUS: s.fabric.EstimateQueueUS(s.clock, s.access[0].WritePath),
		GPUWaitUS: max(0, p.lastGPUReady-s.clock)}
	for _, hash := range s.ready {
		c.HBM[hash]++
	}
	for _, j := range s.fabric.groupBuffer {
		if j.record.Request == request && j.record.Destination == h.pool {
			c.BufferedStoreBlocks += max(1, j.units)
		}
	}
	// Existing staged metadata is known work even before its host submission.
	// Stores precede loads; shared path resources serialize both directions.
	shared := false
	for _, r := range s.access[0].ReadPath {
		for _, w := range s.access[0].WritePath {
			shared = shared || r == w
		}
	}
	hostEnd := int64(0)
	for _, direction := range []string{"d2h", "h2d"} {
		for _, j := range p.staged {
			load := j.record.Destination == "hbm"
			if load != (direction == "h2d") {
				continue
			}
			n := max(1, j.units)
			submit := m.model.HostSubmitUS(direction, n)
			hostEnd += submit
			c.KnownStagedHostUS += submit
			c.KnownStagedJobs++
			if load {
				c.ReadQueueUS = max(c.ReadQueueUS, hostEnd, c.GPUWaitUS) + m.dataUS(direction, n)
				if shared {
					c.WriteQueueUS = max(c.WriteQueueUS, c.ReadQueueUS)
				}
			} else {
				c.WriteQueueUS = max(c.WriteQueueUS, hostEnd, c.GPUWaitUS) + m.dataUS(direction, n)
				if shared {
					c.ReadQueueUS = max(c.ReadQueueUS, c.WriteQueueUS)
				}
			}
		}
	}
	// Pending-copy availability is a current-profile projection. It includes
	// an enclosing engine/poll envelope; unknown future co-batches stay a gap.
	pending := map[string]int64{}
	note := func(j *peerJob, physicalDone, buffered bool) {
		if j.record.Destination != h.pool {
			return
		}
		ready := max(c.WriteQueueUS, c.GPUWaitUS) + m.idlePeriodUS()
		if buffered {
			ready = c.storeReadyUS(max(1, c.BufferedStoreBlocks))
		}
		if physicalDone {
			ready = m.idlePeriodUS()
		}
		if j.record.Name == "transfer_start" {
			ready = max(c.GPUWaitUS, max(0, m.dataUS("d2h", max(1, j.units))-(s.clock-j.record.Time))) + m.idlePeriodUS()
		}
		if j.record.Hash != "" {
			pending[j.record.Hash] = ready
		}
		for _, hash := range j.record.Hashes {
			pending[hash] = ready
		}
	}
	for _, j := range p.jobs {
		note(j.job, j.physicalDone, false)
	}
	for _, j := range s.fabric.groupBuffer {
		note(j, false, true)
	}
	for hash, e := range pool.entries {
		x := PlacementHostCopy{Ready: e.ready, Pinned: e.readers > 0 || !e.ready}
		if !e.ready {
			var ok bool
			x.ReadyInUS, ok = pending[hash]
			if !ok {
				panic("unpublished host copy has no visible transfer for benefit forecast")
			}
		}
		c.Host[hash] = x
	}
	return c
}

func (c *PlacementPhaseCosts) idlePeriodUS() int64 {
	t := c.model.PredictEngineStep(nil)
	return c.decisionUS + t.PostForwardUS + t.PollUS + t.TailUS
}

func (c *PlacementCostSnapshot) storeReadyUS(blocks int64) int64 {
	// Single-reader envelope, including staging until the next engine step,
	// known data/GPU queues, submission, physical copy and later adoption.
	m := c.model
	return m.idlePeriodUS() + max(c.WriteQueueUS, c.GPUWaitUS) + m.storeService(blocks) + m.idlePeriodUS()
}

// access reproduces the native contiguous-prefix rule on a detached copy set.
// The hypothetical query is one token beyond the candidate's known prefix.
func (c *PlacementCostSnapshot) access(path []string, reuseOffset int64, blockTokens int64) PlacementAccessCost {
	hbm := make(map[string]bool, len(c.HBM))
	for h, n := range c.HBM {
		hbm[h] = n > 0
	}
	r := PlacementAccessCost{}
	index := 0
	for index < len(path) {
		if hbm[path[index]] {
			index++
			continue
		}
		end := index
		pending := int64(0)
		for end < len(path) {
			copy, ok := c.Host[path[end]]
			if !ok {
				break
			}
			pending = max(pending, copy.ReadyInUS)
			end++
		}
		if end == index {
			break
		}
		wait := max(0, pending-reuseOffset-r.TotalUS)
		r.TotalUS += wait
		r.PendingStoreWaitUS += wait
		end = min(end, index+int(c.RestoreWindow))
		n := int64(end - index)
		load := c.model.load(n, max(0, c.ReadQueueUS-reuseOffset-r.TotalUS), max(0, c.GPUWaitUS-reuseOffset-r.TotalUS))
		r.TotalUS += load.ReadyUS
		r.LoadReadyUS += load.ReadyUS
		r.LoadPlanningUS += load.PlanningUS
		r.LoadSubmitUS += load.SubmitUS
		r.LoadDataUS += load.DataUS
		r.LoadQueueUS += load.QueueUS
		r.LoadAdoptionWaitUS += load.AdoptionWaitUS
		r.LoadedBlocks += n
		r.LoadGroups++
		for _, hash := range path[index:end] {
			hbm[hash] = true
		}
		index = end
	}
	compute := c.model.compute(int64(index)*blockTokens, int64(len(path)-index)*blockTokens+1)
	r.TotalUS += compute.TotalUS
	r.ComputeUS = compute.ComputeUS
	r.ComputeSteps = compute.ComputeSteps
	r.ComputedTokens = compute.ComputedTokens
	return r
}
