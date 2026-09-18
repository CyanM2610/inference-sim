package kv

import (
	"container/heap"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

type hotPrefixUse struct {
	hash   string
	tokens int64
}

type hotPrefixRequest struct {
	credited      map[string]bool
	shadowMatches map[string]int64 // timely logical re-reference, committed on use/publication
	pending       []hotPrefixUse
}

type hotPrefixShadow struct {
	hash    string
	expires int64
	serial  uint64
}

type hotPrefixExpiries []hotPrefixShadow

func (q hotPrefixExpiries) Len() int { return len(q) }
func (q hotPrefixExpiries) Less(i, j int) bool {
	return q[i].expires < q[j].expires || q[i].expires == q[j].expires && q[i].serial < q[j].serial
}
func (q hotPrefixExpiries) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *hotPrefixExpiries) Push(v any)   { *q = append(*q, v.(hotPrefixShadow)) }
func (q *hotPrefixExpiries) Pop() any {
	v := (*q)[len(*q)-1]
	*q = (*q)[:len(*q)-1]
	return v
}

func (s *PeerCache) hotPrefixHasCopy(hash string) bool {
	// Unpublished STORE targets also preserve the existing heat generation.
	if _, local := s.lookup[hash]; local || s.fabric.pools[s.hotprefix.pool].entries[hash] != nil {
		return true
	}
	// The legacy lookup index can lose a hash when its newest physical copy
	// is removed. Older completed copies still prevent a metadata-only shadow.
	for _, readyHash := range s.ready {
		if readyHash == hash {
			return true
		}
	}
	return false
}

func (s *PeerCache) expireHotPrefixShadows() {
	h := s.hotprefix
	if h == nil {
		return
	}
	for len(h.expiry) > 0 && h.expiry[0].expires <= s.clock {
		x := heap.Pop(&h.expiry).(hotPrefixShadow)
		current, ok := h.shadows[x.hash]
		if !ok || current.serial != x.serial {
			continue
		}
		delete(h.shadows, x.hash)
		if !s.hotPrefixHasCopy(x.hash) {
			s.hotPrefixRecord("hotprefix_shadow_expire", "", x.hash, "")
			n := h.policy.nodes[x.hash]
			n.frequency, n.clock = 0, 0
		}
	}
}

// Called after an actual HBM drop. A declined admission is not by itself a
// shadow: duplicates, ready DRAM copies and in-flight stores still own KV.
func (s *PeerCache) shadowHotPrefixDrop(hash, request string) {
	h := s.hotprefix
	if h == nil || s.hotPrefixHasCopy(hash) {
		return
	}
	n := h.policy.nodes[hash]
	if n == nil || n.frequency == 0 || n.frequency >= h.policy.shadowThreshold() {
		return
	}
	if h.shadowTTL == 0 {
		n.frequency, n.clock = 0, 0
		return
	}
	if h.shadowTTL > math.MaxInt64-s.clock {
		panic("HotPrefix shadow expiry overflows simulation clock")
	}
	h.shadowSerial++
	x := hotPrefixShadow{hash: hash, expires: s.clock + h.shadowTTL, serial: h.shadowSerial}
	h.shadows[hash] = x
	heap.Push(&h.expiry, x)
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "hotprefix_shadow_create", Instance: s.id, Request: request, Hash: hash,
		Reason: "frequency_below_threshold", Counters: map[string]int64{"frequency": n.frequency, "clock": n.clock, "expires_at_us": x.expires}})
}

// Legacy configurations couple low-frequency Shadow eligibility to admission.
// An explicit threshold isolates this mechanism in admission experiments. A
// common warmup must not accidentally use the target's threshold before switch.
func (p *HotPrefixPolicy) shadowThreshold() int64 {
	if p.config.ShadowThreshold != nil {
		return *p.config.ShadowThreshold
	}
	if c := p.config.AdmissionCost; c != nil && c.WarmupUntilFullReady != nil && !p.admissionWarmupFinished {
		return c.WarmupUntilFullReady.AdmissionThreshold
	}
	return p.config.AdmissionThreshold
}

// Use the same effective original-prefix range as cache reuse metrics, but
// retain block identities for the policy. Neither allocation nor DMA completion
// proves that the request executed with those blocks.
func (s *PeerCache) stageHotPrefixReuse(req *sim.Request, start int64, cached []int64, fresh bool) {
	if s.hotprefix == nil || !fresh {
		return
	}
	r := s.hotprefix.requests[req.ID]
	r.pending = nil
	remaining := min(start, max(int64(0), req.InputLen()-1))
	for _, id := range cached {
		n := min(s.BlockSizeTokens, remaining)
		if n <= 0 {
			break
		}
		remaining -= n
		r.pending = append(r.pending, hotPrefixUse{hash: s.Blocks[id].Hash, tokens: n})
	}
}

func (s *PeerCache) completeHotPrefixReuse(req *sim.Request) {
	h := s.hotprefix
	if h == nil {
		return
	}
	s.expireHotPrefixShadows()
	r := h.requests[req.ID]
	if r == nil {
		return
	}
	for _, use := range r.pending {
		if r.credited[use.hash] {
			continue
		}
		s.completedBenefitReference(use.hash, req.ID)
		h.policy.reuse(use.hash, r.shadowMatches[use.hash])
		if _, matched := r.shadowMatches[use.hash]; matched {
			s.hotPrefixRecord("hotprefix_shadow_reuse", req.ID, use.hash, "", "kv_reused")
			delete(r.shadowMatches, use.hash)
		}
		r.credited[use.hash] = true
		s.consumeHotPrefixFuture(use.hash, req.ID)
		n := h.policy.nodes[use.hash]
		s.fabric.emit(PeerRecord{Time: s.clock, Name: "hotprefix_reuse", Instance: s.id, Request: req.ID, Hash: use.hash,
			Reason: "compute_completed", Counters: map[string]int64{"frequency": n.frequency, "clock": n.clock, "reused_tokens": use.tokens}})
	}
	r.pending = nil
}

// Each new physical compute publication contributes at most one initial use.
// Repeated MirrorToCPU calls and the request's own recovery are idempotent.
func (s *PeerCache) publishHotPrefixCompute(req *sim.Request, b *KVBlock) {
	h := s.hotprefix
	if h == nil {
		return
	}
	r := h.requests[req.ID]
	if r.credited[b.Hash] {
		return
	}
	s.completedBenefitReference(b.Hash, req.ID)
	if frequency, matched := r.shadowMatches[b.Hash]; matched {
		h.policy.reuse(b.Hash, frequency)
		s.hotPrefixRecord("hotprefix_shadow_reuse", req.ID, b.Hash, "", "recomputed")
		delete(r.shadowMatches, b.Hash)
	} else {
		h.policy.published(b.Hash)
	}
	r.credited[b.Hash] = true
	s.consumeHotPrefixFuture(b.Hash, req.ID)
}
