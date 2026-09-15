package policylab

import "github.com/inference-sim/inference-sim/sim/kv"

type CacheMetrics struct {
	Schema   string                   `json:"schema"`
	Scope    string                   `json:"scope"`
	Totals   kv.CacheMetricsSummary   `json:"totals"`
	Requests []kv.CacheRequestMetrics `json:"requests"`
}

func collectCacheMetrics(c Config, stores []*kv.PeerCache) CacheMetrics {
	byID := map[string]kv.CacheRequestMetrics{}
	for _, store := range stores {
		for _, r := range store.CacheRequestStatistics() {
			byID[r.Request] = r
		}
	}
	m := CacheMetrics{Schema: "prefix_cache_sources_v1", Scope: "original_prompt_first_lookup_and_first_completed_compute; pool_id_dram_is_DRAM; token_rates_include_required_replay_and_uncacheable_tail_in_denominator"}
	for _, request := range c.Requests {
		r, ok := byID[request.ID]
		if !ok {
			r = kv.CacheRequestMetrics{Request: request.ID, InputTokens: int64(len(request.Input))}
		}
		m.Requests = append(m.Requests, r)
	}
	m.Totals = kv.SummarizeCacheRequests(m.Requests)
	return m
}
