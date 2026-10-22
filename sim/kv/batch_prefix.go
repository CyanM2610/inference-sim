package kv

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

type batchPrefixBlock struct {
	block, index   int64
	hash, producer string
	request        *sim.Request
}
type batchPrefixState struct {
	planning  bool
	offers    map[string]batchPrefixBlock
	claims    map[string][]batchPrefixBlock
	submitted []batchPrefixBlock
}

func (s *PeerCache) BatchPrefixReuseEnabled() bool { return s.batchPrefixes != nil }

// EnableBatchPrefixReuse covers the single abstract engine-phase backend, whose
// one submitted batch executes all layer KV writes before that layer's attention.
// It does not mark whole-model KV ready during scheduling or invent layer timing.
func (s *PeerCache) EnableBatchPrefixReuse() error {
	if s.fabric.phases == nil || s.fabric.phases.active || s.fabric.Pending() != 0 || s.batchPrefixes != nil {
		return fmt.Errorf("batch prefix reuse requires idle single-instance engine phases")
	}
	s.batchPrefixes = &batchPrefixState{}
	return nil
}

func (s *PeerCache) BeginPrefixBatch() bool {
	p := s.batchPrefixes
	if p == nil {
		return false
	}
	if p.planning || len(p.submitted) > 0 {
		panic("overlapping batch prefix scopes")
	}
	p.planning = true
	p.offers = map[string]batchPrefixBlock{}
	p.claims = map[string][]batchPrefixBlock{}
	return true
}

func (s *PeerCache) OfferPrefixWork(w sim.BatchWork) {
	p := s.batchPrefixes
	if p == nil || !p.planning {
		panic("prefix work outside formation")
	}
	end := min(w.PrefixTokens+w.NewTokens, w.Request.InputLen())
	for i, bid := range s.RequestMap[w.Request.ID] {
		blockEnd := int64(i+1) * s.BlockSizeTokens
		if blockEnd <= w.PrefixTokens || blockEnd > end {
			continue
		}
		b := s.Blocks[bid]
		if b.Hash == "" || int64(len(b.Tokens)) != s.BlockSizeTokens {
			panic("offered prefix lacks complete allocated input")
		}
		if _, exists := p.offers[b.Hash]; !exists {
			p.offers[b.Hash] = batchPrefixBlock{bid, int64(i), b.Hash, w.Request.ID, w.Request}
		}
	}
}

func (s *PeerCache) validPrefixDonor(b batchPrefixBlock) bool {
	ids := s.RequestMap[b.producer]
	return b.request.NumNewTokens > 0 && b.request.State == sim.StateRunning && b.index < int64(len(ids)) && ids[b.index] == b.block && s.Blocks[b.block].Hash == b.hash && s.Blocks[b.block].RefCount > 0
}

func (s *PeerCache) batchCachedPrefix(req *sim.Request) sim.CachedPrefix {
	p := s.batchPrefixes
	keys := s.fullPrefixHashes(req.FullInputTokens())
	base := s.localPrefix(req.ID, req.InputLen(), keys)
	// Existing restore receipts retain their own range. Only fresh block tables
	// may acquire another producer's in-batch prefix, just as in native admission.
	if len(s.RequestMap[req.ID]) > 0 || s.readPending[req.ID] > 0 || len(s.holds[req.ID]) > 0 {
		return base
	}
	end := min(int64(len(keys)), max(int64(0), (req.InputLen()-1)/s.BlockSizeTokens))
	if int64(len(base.Blocks)) >= end {
		return base
	}
	var blocks []int64
	var claims []batchPrefixBlock
	for _, key := range keys[:end] {
		if bid, ok := s.lookup[key]; ok {
			blocks = append(blocks, bid)
			continue
		}
		donor, ok := p.offers[key]
		if !ok || donor.producer == req.ID || !s.validPrefixDonor(donor) {
			break
		}
		blocks = append(blocks, donor.block)
		claims = append(claims, donor)
	}
	p.claims[req.ID] = claims
	return sim.CachedPrefix{Blocks: blocks, Tokens: int64(len(blocks)) * s.BlockSizeTokens}
}

func (s *PeerCache) CommitPrefixBatch(work []sim.BatchWork) {
	p := s.batchPrefixes
	if p == nil || !p.planning {
		panic("prefix commit outside formation")
	}
	actual := map[string]sim.BatchWork{}
	for _, w := range work {
		actual[w.Request.ID] = w
	}
	type claim struct {
		consumer string
		block    batchPrefixBlock
	}
	var accepted []claim
	consumers := make([]string, 0, len(p.claims))
	for id := range p.claims {
		consumers = append(consumers, id)
	}
	sort.Strings(consumers)
	for _, consumer := range consumers {
		owned := map[int64]bool{}
		for _, bid := range s.RequestMap[consumer] {
			owned[bid] = true
		}
		for _, bid := range s.holds[consumer] {
			owned[bid] = true
		}
		for _, b := range p.claims[consumer] {
			if !owned[b.block] {
				continue
			} // allocation failed without taking the hit
			w, ok := actual[b.producer]
			end := (b.index + 1) * s.BlockSizeTokens
			if !ok || !s.validPrefixDonor(b) || w.PrefixTokens >= end || w.PrefixTokens+w.NewTokens < end {
				panic("batch consumer references removed or truncated producer")
			}
			accepted = append(accepted, claim{consumer, b})
		}
	}
	// Validate every reference before creating submitted-work leases or events.
	leased := map[int64]bool{}
	for _, c := range accepted {
		b := c.block
		w := actual[b.producer]
		if !leased[b.block] {
			s.pin(s.Blocks[b.block])
			p.submitted = append(p.submitted, b)
			leased[b.block] = true
		}
		s.fabric.emit(PeerRecord{Time: s.clock, Name: "batch_prefix_dependency", Instance: s.id, Request: c.consumer, Requests: []string{b.producer}, Hash: b.hash, HBMBlocks: []int64{b.block},
			Counters: map[string]int64{"logical_index": b.index, "producer_prefix_tokens": w.PrefixTokens, "producer_new_tokens": w.NewTokens}})
	}
}

func (s *PeerCache) AbortPrefixBatch() {
	if p := s.batchPrefixes; p != nil {
		p.planning = false
		p.offers = nil
		p.claims = nil
	}
}

func (s *PeerCache) CompletePrefixBatch() {
	p := s.batchPrefixes
	if p == nil {
		return
	}
	if p.planning {
		panic("batch completed while prefix planning remains open")
	}
	for _, donor := range p.submitted {
		b := s.Blocks[donor.block]
		if b.Hash != donor.hash || b.RefCount <= 0 {
			panic("submitted prefix overwritten before completion")
		}
		// Submitted computation still produces these bytes if a request cancels.
		// The donation lease prevents reuse, and consumers retain their own refs.
		s.publishReadyCopy(b)
		s.fabric.emit(PeerRecord{Time: s.clock, Name: "batch_prefix_published", Instance: s.id, Request: donor.producer, Hash: b.Hash, HBMBlocks: []int64{b.ID}})
		s.unpin(b)
	}
	p.submitted = nil
}
