package kv

import (
	"fmt"
	"math"
)

// HotPrefixBenefitConfig separates the forecast from historical APC frequency.
// The first experiment predicts the next completed reference in a fixed window.
type HotPrefixBenefitConfig struct {
	HorizonUS int64 `json:"horizon_us"`
	DecayUS   int64 `json:"decay_us"`
}

func (c HotPrefixBenefitConfig) Validate() error {
	if c.HorizonUS <= 0 || c.DecayUS <= 0 || c.HorizonUS > 1<<60 || c.DecayUS > 1<<60 {
		return fmt.Errorf("benefit forecast requires positive bounded horizon/decay")
	}
	return nil
}

type hotPrefixReferenceHistory struct {
	weight       float64
	lastUS       int64
	observations int64
}

type HotPrefixForecast struct {
	ObservedReferences int64   `json:"observed_references"`
	LastReferenceUS    int64   `json:"last_reference_us"`
	ObservedThroughUS  int64   `json:"observed_through_us"`
	HorizonUS          int64   `json:"horizon_us"`
	DecayUS            int64   `json:"decay_us"`
	Weight             float64 `json:"decayed_weight"`
	ExpectedCount      float64 `json:"expected_count"`
	NextProbability    float64 `json:"next_reference_probability"`
	NextOffsetUS       int64   `json:"conditional_mean_next_offset_us"`
}

func (h *hotPrefixReferenceHistory) completed(now int64, c HotPrefixBenefitConfig) {
	if now < 0 || h.observations > 0 && now < h.lastUS {
		panic("completed reference clock regressed")
	}
	if h.observations > 0 {
		h.weight *= math.Exp(-float64(now-h.lastUS) / float64(c.DecayUS))
	}
	h.weight++
	h.lastUS = now
	h.observations++
}

func (h hotPrefixReferenceHistory) predict(now int64, c HotPrefixBenefitConfig) HotPrefixForecast {
	if now < 0 || h.observations > 0 && now < h.lastUS {
		panic("forecast precedes observed reference")
	}
	f := HotPrefixForecast{ObservedReferences: h.observations, LastReferenceUS: h.lastUS,
		ObservedThroughUS: now, HorizonUS: c.HorizonUS, DecayUS: c.DecayUS}
	if h.observations == 0 {
		f.NextOffsetUS = c.HorizonUS / 2
		return f
	}
	f.Weight = h.weight * math.Exp(-float64(now-h.lastUS)/float64(c.DecayUS))
	f.ExpectedCount = f.Weight * float64(c.HorizonUS) / float64(c.DecayUS)
	f.NextProbability = -math.Expm1(-f.ExpectedCount)
	// E[T | T <= H] for an exponential inter-reference model. The series
	// avoids catastrophic cancellation when an old history has almost decayed.
	mu := f.ExpectedCount
	fraction := .5 - mu/12
	if mu > 1e-4 {
		fraction = 1/mu - 1/math.Expm1(mu)
	}
	f.NextOffsetUS = min(c.HorizonUS, max(0, int64(math.Round(float64(c.HorizonUS)*fraction))))
	return f
}

func (s *PeerCache) completedBenefitReference(hash, request string) {
	h := s.hotprefix
	if h == nil || h.policy.config.Benefit == nil {
		return
	}
	started := h.policy.cpuStart()
	n := h.policy.nodes[hash]
	if n == nil {
		panic("benefit reference lacks observed prefix identity")
	}
	n.references.completed(s.clock, *h.policy.config.Benefit)
	h.policy.cpuEnd("reference_history", started)
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "hotprefix_reference_completed", Instance: s.id,
		Request: request, Hash: hash, Reason: "once_per_completed_request_hash", Counters: map[string]int64{"observations": n.references.observations}})
}
