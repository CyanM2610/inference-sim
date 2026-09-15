package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"sort"
)

// CacheTokenCounts measures original prompt tokens, never decode allocations or
// repeated recovery. A full restored final block saves at most input_len-1 tokens.
type CacheTokenCounts struct {
	HBMHitTokens               int64 `json:"hbm_hit_tokens"`
	DRAMHitTokens              int64 `json:"dram_hit_tokens"`
	OtherPoolHitTokens         int64 `json:"other_pool_hit_tokens"`
	HBMReusedTokens            int64 `json:"hbm_reused_tokens"`
	HBMPrefetchedReusedTokens  int64 `json:"hbm_prefetched_reused_tokens"` // subset of HBM
	DRAMReusedTokens           int64 `json:"dram_reused_tokens"`
	DRAMDemandReusedTokens     int64 `json:"dram_demand_reused_tokens"`
	DRAMPromotionReusedTokens  int64 `json:"dram_promotion_reused_tokens"`
	DRAMSharedLoadReusedTokens int64 `json:"dram_shared_load_reused_tokens"` // subset of DRAM
	DRAMInitialHitReusedTokens int64 `json:"dram_initial_hit_reused_tokens"`
	OtherPoolReusedTokens      int64 `json:"other_pool_reused_tokens"`
	ReadyAfterWaitReusedTokens int64 `json:"ready_after_wait_reused_tokens"`
	BatchSharedReusedTokens    int64 `json:"batch_shared_reused_tokens"`
	ReusedTokens               int64 `json:"reused_tokens"`
}

// CacheRequestMetrics keeps availability and actual use separate. Lookup is the
// first allocation or request-promotion attempt, before its policy loading cap.
// Reuse is committed only by the first completed compute, not allocation or DMA.
type CacheRequestMetrics struct {
	Request            string `json:"request"`
	Instance           string `json:"instance"`
	InputTokens        int64  `json:"input_tokens"`
	LookupObserved     bool   `json:"lookup_observed"`
	LookupUS           int64  `json:"lookup_us"`
	LookupPendingStore bool   `json:"lookup_pending_store"`
	ReuseObserved      bool   `json:"reuse_observed"`
	ReuseUS            int64  `json:"reuse_us"`
	CacheTokenCounts
}

type cacheOrigin struct {
	serial              int64
	pool, reason, owner string
}

type cacheRequestObservation struct {
	CacheRequestMetrics
	serial   int64
	dramHits map[int64]int64 // logical index -> initially available token count
	pending  *CacheTokenCounts
}

type cacheMetricState struct {
	serial   int64
	origins  map[int64]cacheOrigin
	requests map[string]*cacheRequestObservation
}

func (s *PeerCache) metricState() *cacheMetricState {
	if s.cacheMetrics == nil {
		s.cacheMetrics = &cacheMetricState{origins: map[int64]cacheOrigin{}, requests: map[string]*cacheRequestObservation{}}
	}
	return s.cacheMetrics
}

func (s *PeerCache) metricPublish(b *KVBlock) {
	m := s.metricState()
	if _, ok := m.origins[b.ID]; !ok {
		m.serial++
		m.origins[b.ID] = cacheOrigin{serial: m.serial}
	}
}

func (s *PeerCache) metricLoad(b *KVBlock, pool, reason, owner string) {
	m := s.metricState()
	o := m.origins[b.ID]
	o.pool, o.reason, o.owner = pool, reason, owner
	m.origins[b.ID] = o
}

// Metrics hashing deliberately bypasses simulated policy HashWork accounting.
// This observer never touches heat, LRU, readiness, pins or a scheduling query.
func (s *PeerCache) observeCacheLookup(id string, input []sim.TokenID) {
	m := s.metricState()
	if _, ok := m.requests[id]; ok {
		return
	}
	n := int64(len(input))
	r := &cacheRequestObservation{CacheRequestMetrics: CacheRequestMetrics{
		Request: id, Instance: s.id, InputTokens: n, LookupObserved: true, LookupUS: s.clock,
	}, serial: m.serial, dramHits: map[int64]int64{}}
	m.requests[id] = r
	previous := ""
	local := true
	for i := int64(0); (i+1)*s.BlockSizeTokens <= n; i++ {
		previous = hash.HashBlock(previous, input[i*s.BlockSizeTokens:(i+1)*s.BlockSizeTokens])
		tokens := min(s.BlockSizeTokens, max(int64(0), n-1-i*s.BlockSizeTokens))
		// Plain APC excludes the last complete prompt block; restore control can
		// load it and replay its last token, which is not a saved token.
		if local && (i+1)*s.BlockSizeTokens < n {
			if _, ok := s.lookup[previous]; ok {
				r.HBMHitTokens += tokens
				continue
			}
		}
		local = false
		if !s.restoreControl && (i+1)*s.BlockSizeTokens >= n {
			break
		}
		if e, a := s.find(previous); e != nil {
			if a.Pool == "dram" {
				r.DRAMHitTokens += tokens
				r.dramHits[i] = tokens
			} else {
				r.OtherPoolHitTokens += tokens
			}
			continue
		}
		for _, a := range s.access {
			if e := s.fabric.pools[a.Pool].entries[previous]; e != nil && !e.ready {
				r.LookupPendingStore = true
			}
		}
		break // only a contiguous accessible prefix is a hit
	}
}

// stageCacheReuse runs after successful allocation, while cached[] still
// describes the prefix actually accepted. Release before compute discards it.
func (s *PeerCache) stageCacheReuse(req *sim.Request, start int64, cached []int64, fresh bool) {
	r := s.metricState().requests[req.ID]
	if r == nil || r.ReuseObserved || !fresh {
		return
	}
	c := CacheTokenCounts{}
	remaining := min(start, max(int64(0), r.InputTokens-1))
	for index, id := range cached {
		n := min(s.BlockSizeTokens, remaining)
		if n <= 0 {
			break
		}
		remaining -= n
		c.ReusedTokens += n
		o, ready := s.metricState().origins[id]
		switch {
		case !ready:
			c.BatchSharedReusedTokens += n
		case o.serial <= r.serial:
			c.HBMReusedTokens += n
			if o.reason == "promotion" {
				c.HBMPrefetchedReusedTokens += n
			}
		case o.pool == "dram":
			c.DRAMReusedTokens += n
			if o.reason == "promotion" {
				c.DRAMPromotionReusedTokens += n
			} else {
				c.DRAMDemandReusedTokens += n
			}
			if o.owner != req.ID {
				c.DRAMSharedLoadReusedTokens += n
			}
			c.DRAMInitialHitReusedTokens += min(n, r.dramHits[int64(index)])
		case o.pool != "":
			c.OtherPoolReusedTokens += n
		default:
			c.ReadyAfterWaitReusedTokens += n
		}
	}
	r.pending = &c
}

func (s *PeerCache) completeCacheReuse(req *sim.Request) {
	r := s.metricState().requests[req.ID]
	if r == nil || r.ReuseObserved || r.pending == nil {
		return
	}
	hbm, dram, other := r.HBMHitTokens, r.DRAMHitTokens, r.OtherPoolHitTokens
	r.CacheTokenCounts = *r.pending
	r.HBMHitTokens, r.DRAMHitTokens, r.OtherPoolHitTokens = hbm, dram, other
	r.ReuseObserved, r.ReuseUS = true, s.clock
	r.pending, r.dramHits = nil, nil
}

// CacheRequestStatistics returns detached rows; neither reads nor retries count
// as another request. Physical transfers remain in the existing event metrics.
func (s *PeerCache) CacheRequestStatistics() []CacheRequestMetrics {
	if s.cacheMetrics == nil {
		return nil
	}
	var rows []CacheRequestMetrics
	var ids []string
	for id := range s.cacheMetrics.requests {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		rows = append(rows, s.cacheMetrics.requests[id].CacheRequestMetrics)
	}
	return rows
}

type CacheMetricsSummary struct {
	Requests              int64 `json:"requests"`
	LookupRequests        int64 `json:"lookup_requests"`
	ReuseObservedRequests int64 `json:"reuse_observed_requests"`
	LookupInputTokens     int64 `json:"lookup_input_tokens"`
	ReuseInputTokens      int64 `json:"reuse_input_tokens"`
	HBMHitRequests        int64 `json:"hbm_hit_requests"`
	DRAMHitRequests       int64 `json:"dram_hit_requests"`
	DRAMReusedRequests    int64 `json:"dram_reused_requests"`
	PendingStoreRequests  int64 `json:"pending_store_requests"`
	CacheTokenCounts
	HBMDirectHitRate            *float64 `json:"hbm_direct_hit_rate"`
	DRAMHitRate                 *float64 `json:"dram_hit_rate"`
	DRAMConditionalHitRate      *float64 `json:"dram_conditional_hit_rate"`
	HBMReuseRate                *float64 `json:"hbm_reuse_rate"`
	DRAMReuseRate               *float64 `json:"dram_reuse_rate"`
	DRAMInitialHitReuseFraction *float64 `json:"dram_initial_hit_reuse_fraction"`
	PrefixReuseRate             *float64 `json:"prefix_reuse_rate"`
	HBMRequestHitRate           *float64 `json:"hbm_request_hit_rate"`
	DRAMRequestHitRate          *float64 `json:"dram_request_hit_rate"`
	DRAMRequestReuseRate        *float64 `json:"dram_request_reuse_rate"`
}

func cacheRatio(n, d int64) *float64 {
	if d == 0 {
		return nil
	}
	x := float64(n) / float64(d)
	return &x
}

// SummarizeCacheRequests sums token counts before dividing, so long and short
// requests are weighted by demand rather than averaged as per-request rates.
func SummarizeCacheRequests(rows []CacheRequestMetrics) CacheMetricsSummary {
	s := CacheMetricsSummary{Requests: int64(len(rows))}
	for _, r := range rows {
		if r.LookupObserved {
			s.LookupRequests++
			s.LookupInputTokens += r.InputTokens
			if r.HBMHitTokens > 0 {
				s.HBMHitRequests++
			}
			if r.DRAMHitTokens > 0 {
				s.DRAMHitRequests++
			}
			if r.LookupPendingStore {
				s.PendingStoreRequests++
			}
		}
		if r.ReuseObserved {
			s.ReuseObservedRequests++
			s.ReuseInputTokens += r.InputTokens
			if r.DRAMReusedTokens > 0 {
				s.DRAMReusedRequests++
			}
		}
		s.HBMHitTokens += r.HBMHitTokens
		s.DRAMHitTokens += r.DRAMHitTokens
		s.OtherPoolHitTokens += r.OtherPoolHitTokens
		s.HBMReusedTokens += r.HBMReusedTokens
		s.HBMPrefetchedReusedTokens += r.HBMPrefetchedReusedTokens
		s.DRAMReusedTokens += r.DRAMReusedTokens
		s.DRAMDemandReusedTokens += r.DRAMDemandReusedTokens
		s.DRAMPromotionReusedTokens += r.DRAMPromotionReusedTokens
		s.DRAMSharedLoadReusedTokens += r.DRAMSharedLoadReusedTokens
		s.DRAMInitialHitReusedTokens += r.DRAMInitialHitReusedTokens
		s.OtherPoolReusedTokens += r.OtherPoolReusedTokens
		s.ReadyAfterWaitReusedTokens += r.ReadyAfterWaitReusedTokens
		s.BatchSharedReusedTokens += r.BatchSharedReusedTokens
		s.ReusedTokens += r.ReusedTokens
	}
	s.HBMDirectHitRate = cacheRatio(s.HBMHitTokens, s.LookupInputTokens)
	s.DRAMHitRate = cacheRatio(s.DRAMHitTokens, s.LookupInputTokens)
	s.DRAMConditionalHitRate = cacheRatio(s.DRAMHitTokens, s.LookupInputTokens-s.HBMHitTokens)
	s.HBMReuseRate = cacheRatio(s.HBMReusedTokens, s.ReuseInputTokens)
	s.DRAMReuseRate = cacheRatio(s.DRAMReusedTokens, s.ReuseInputTokens)
	s.DRAMInitialHitReuseFraction = cacheRatio(s.DRAMInitialHitReusedTokens, s.DRAMHitTokens)
	s.PrefixReuseRate = cacheRatio(s.ReusedTokens, s.ReuseInputTokens)
	s.HBMRequestHitRate = cacheRatio(s.HBMHitRequests, s.LookupRequests)
	s.DRAMRequestHitRate = cacheRatio(s.DRAMHitRequests, s.LookupRequests)
	s.DRAMRequestReuseRate = cacheRatio(s.DRAMReusedRequests, s.ReuseObservedRequests)
	return s
}
