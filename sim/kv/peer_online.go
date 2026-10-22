package kv

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

type peerHeat struct {
	tokens []sim.TokenID
	count  float64
	at     int64
}

func (s *PeerCache) heatAt(h *peerHeat, now int64) float64 {
	p := s.fabric.mechanisms.Online
	if p.HalfLifeUS == 0 {
		return h.count
	}
	return h.count * math.Exp2(-float64(now-h.at)/float64(p.HalfLifeUS))
}
func (s *PeerCache) observeHeat(tokens []sim.TokenID) {
	if s.heat == nil {
		return
	}
	for i, h := range s.hashes(tokens) {
		entry := s.heat[h]
		if entry == nil {
			entry = &peerHeat{tokens: tokens[:int64(i+1)*s.BlockSizeTokens+1], at: s.clock}
			s.heat[h] = entry
		}
		entry.count = s.heatAt(entry, s.clock) + 1
		entry.at = s.clock
	}
}

type peerOnlineEvent struct {
	at       int64
	s        *PeerCache
	credit   float64
	previous int64
}

func (e *peerOnlineEvent) Timestamp() int64 { return e.at }
func (e *peerOnlineEvent) Priority() int    { return sim.PriorityAdapterLoad }
func (s *PeerCache) StartOnlinePromotion() {
	if p := s.fabric.mechanisms.Online; p != nil {
		s.schedule(&peerOnlineEvent{at: p.PeriodUS, s: s, credit: float64(int64(p.BurstBlocks) * s.fabric.bytes), previous: p.PeriodUS})
	}
}
func (e *peerOnlineEvent) Execute(_ *sim.Simulator) {
	if cost := e.s.fabric.mechanisms.PolicyUS; cost > 0 {
		e.s.fabric.controlSubmit(e.at, &peerControlJob{record: PeerRecord{Instance: e.s.id, Request: "@promotion_policy", Reason: "promotion_policy"}, service: cost, schedule: e.s.schedule, done: e.decide})
		return
	}
	e.decide(e.at)
}
func (e *peerOnlineEvent) decide(now int64) {
	s := e.s
	s.clock = now
	p := s.fabric.mechanisms.Online
	if now > p.EndUS {
		return
	}
	e.credit = math.Min(float64(int64(p.BurstBlocks)*s.fabric.bytes), e.credit+float64(now-e.previous)*float64(p.BytesPerSecond)/1e6)
	budget := min(p.BlocksPerTick, int(e.credit/float64(s.fabric.bytes)))
	type candidate struct {
		hash  string
		entry *peerHeat
		score float64
	}
	var choices []candidate
	for h, v := range s.heat {
		score := s.heatAt(v, now)
		if _, resident := s.lookup[h]; !resident && !s.restoring[h] && score >= p.MinFrequency {
			choices = append(choices, candidate{h, v, score})
		}
	}
	sort.Slice(choices, func(i, j int) bool {
		a, b := choices[i], choices[j]
		if a.score != b.score {
			return a.score > b.score
		}
		if len(a.entry.tokens) != len(b.entry.tokens) {
			return len(a.entry.tokens) < len(b.entry.tokens)
		}
		return a.hash < b.hash
	})
	count := 0
	attempted := map[string]bool{}
	for _, c := range choices {
		if count == budget {
			break
		}
		count += s.promotePrefix(c.entry.tokens, budget-count, attempted)
	}
	e.credit -= float64(int64(count) * s.fabric.bytes)
	s.fabric.emit(PeerRecord{Time: now, Name: "online_promotion_tick", Instance: s.id, Value: int64(count), Counters: map[string]int64{"candidate_blocks": int64(len(choices)), "submitted_blocks": int64(count), "credit_bytes": int64(e.credit)}})
	if now+p.PeriodUS <= p.EndUS {
		s.schedule(&peerOnlineEvent{at: now + p.PeriodUS, s: s, credit: e.credit, previous: now})
	}
}
