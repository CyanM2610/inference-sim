package kv

// HotPrefixAdmissionConfig selects the D gate or records the same cost
// prediction while retaining the original frequency threshold baseline.
type HotPrefixAdmissionConfig struct {
	Rule                 string                          `json:"rule"`
	WarmupUntilFullReady *HotPrefixAdmissionWarmupConfig `json:"warmup_until_full_ready,omitempty"`
}

type HotPrefixAdmissionEstimate struct {
	Phase               string               `json:"phase,omitempty"`
	ActiveThreshold     *int64               `json:"active_threshold,omitempty"`
	WarmupStateSHA256   string               `json:"warmup_state_sha256,omitempty"`
	WarmupReadyBlocks   int64                `json:"warmup_ready_blocks,omitempty"`
	Rule                string               `json:"rule"`
	Forecast            HotPrefixForecast    `json:"forecast"`
	VictimForecast      *HotPrefixForecast   `json:"victim_forecast,omitempty"`
	Victim              string               `json:"victim,omitempty"`
	PoolFull            bool                 `json:"pool_full"`
	HostUsedBlocks      int64                `json:"host_used_blocks"`
	HostCapacityBlocks  int64                `json:"host_capacity_blocks"`
	BufferedStoreBlocks int64                `json:"buffered_store_blocks"`
	StoreReadyInUS      int64                `json:"store_ready_in_us"`
	Drop                PlacementAccessCost  `json:"drop"`
	Store               PlacementAccessCost  `json:"store"`
	VictimKeep          *PlacementAccessCost `json:"victim_keep,omitempty"`
	VictimDrop          *PlacementAccessCost `json:"victim_drop,omitempty"`
	CurrentStoreUS      int64                `json:"current_incremental_store_us"`
	ExpectedGainUS      float64              `json:"expected_gain_us"`
	OccupancyUS         float64              `json:"occupancy_us"`
	CostAccept          bool                 `json:"cost_accept"`
	CostReason          string               `json:"cost_reason"`
	Coverage            string               `json:"coverage"`
}

func (p *HotPrefixPolicy) assessAdmission(c PoolAdmissionContext) *HotPrefixAdmissionEstimate {
	started := p.cpuStart()
	defer p.cpuEnd("admission_cost_estimate", started)
	s := c.Costs
	if s == nil || s.model == nil || s.HBM[c.Hash] < 1 || s.HostCapacity != c.CapacityBlocks || int64(len(s.Host)) != c.UsedBlocks {
		panic("admission cost lacks a consistent current placement snapshot")
	}
	if _, exists := s.Host[c.Hash]; exists {
		panic("existing host copy must bypass new admission")
	}
	n := p.nodes[c.Hash]
	if n == nil {
		panic("admission cost lacks observed prefix identity")
	}
	e := &HotPrefixAdmissionEstimate{Rule: p.config.AdmissionCost.Rule,
		Forecast: n.references.predict(s.NowUS, *p.config.Benefit), PoolFull: c.UsedBlocks >= c.CapacityBlocks,
		HostUsedBlocks: c.UsedBlocks, HostCapacityBlocks: c.CapacityBlocks, BufferedStoreBlocks: s.BufferedStoreBlocks,
		CurrentStoreUS: s.model.storeService(s.BufferedStoreBlocks+1) - s.model.storeService(s.BufferedStoreBlocks),
		StoreReadyInUS: s.storeReadyUS(s.BufferedStoreBlocks + 1),
		Coverage:       "single_block_next_reference_conditional_prefix_plus_one_query; fixed_HP_victim; one_slot_opportunity_cost_not_all_descendant_future_traffic; current_vector_store_service; unknown_future_batches_reclaims_queues_not_predicted; not_native_validation"}
	dropped := s.clone()
	dropped.HBM[c.Hash]-- // physical action; another READY duplicate may remain
	if e.PoolFull {
		e.Victim = (hotPrefixPoolPolicy{state: p}).Victim(c.Candidates)
		if e.Victim == "" {
			e.CostReason = "no_evictable_copy"
			return e
		}
		copy, ok := dropped.Host[e.Victim]
		if !ok || !copy.Ready || copy.Pinned {
			panic("admission cost replacement is not a legal READY host copy")
		}
		delete(dropped.Host, e.Victim)
	}
	stored := dropped.clone()
	stored.Host[c.Hash] = PlacementHostCopy{Pinned: true, ReadyInUS: e.StoreReadyInUS}
	path := p.prefixPath(c.Hash)
	e.Drop = dropped.access(path, e.Forecast.NextOffsetUS, p.blockTokens)
	e.Store = stored.access(path, e.Forecast.NextOffsetUS, p.blockTokens)
	e.ExpectedGainUS = e.Forecast.NextProbability * float64(max(0, e.Drop.TotalUS-e.Store.TotalUS))
	if e.Victim != "" {
		v := p.nodes[e.Victim]
		if v == nil {
			panic("opportunity cost lacks victim history")
		}
		forecast := v.references.predict(s.NowUS, *p.config.Benefit)
		e.VictimForecast = &forecast
		// Compare one slot with the incoming copy present in both states. This
		// avoids valuing an isolated suffix until its ancestor can be restored.
		kept := stored.clone()
		kept.Host[e.Victim] = s.Host[e.Victim]
		vp := p.prefixPath(e.Victim)
		keepCost := kept.access(vp, forecast.NextOffsetUS, p.blockTokens)
		dropCost := stored.access(vp, forecast.NextOffsetUS, p.blockTokens)
		e.VictimKeep, e.VictimDrop = &keepCost, &dropCost
		e.OccupancyUS = forecast.NextProbability * float64(max(0, dropCost.TotalUS-keepCost.TotalUS))
	}
	e.CostAccept = e.ExpectedGainUS > float64(e.CurrentStoreUS)+e.OccupancyUS
	if e.CostAccept {
		e.CostReason = "cost_benefit_exceeds_store_and_occupancy"
	} else {
		e.CostReason = "cost_benefit_not_above_store_and_occupancy"
	}
	return e
}
