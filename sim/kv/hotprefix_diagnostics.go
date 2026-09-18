package kv

import "time"

// Diagnostics never enter scheduler decisions or simulated service time.
type HotPrefixDiagnosticsConfig struct {
	TraceCandidates bool `json:"trace_candidates"`
	MeasureCPU      bool `json:"measure_cpu"`
}

type HotPrefixCPU struct {
	Calls       int64 `json:"calls"`
	Nanoseconds int64 `json:"nanoseconds"`
}

type HotPrefixChoiceCandidate struct {
	BlockID               int64    `json:"block_id"`
	Hash                  string   `json:"hash"`
	LRUIndex              int      `json:"lru_index"`
	Eligible              bool     `json:"eligible"`
	HasResidentDescendant bool     `json:"has_resident_descendant"`
	Copies                int      `json:"resident_copies"`
	ReadyCopies           []string `json:"ready_copies,omitempty"`
	Frequency             int64    `json:"frequency"`
	Clock                 int64    `json:"clock"`
	LengthTokens          int64    `json:"length_tokens"`
	Depth                 int64    `json:"depth"`
	Score                 float64  `json:"score"`
}

type HotPrefixChoice struct {
	Sequence      int64                      `json:"sequence"`
	TimeUS        int64                      `json:"time_us"`
	Request       string                     `json:"request"`
	DeficitBlocks int64                      `json:"deficit_blocks"`
	ScoreRule     string                     `json:"score_rule"`
	CandidateRule string                     `json:"candidate_rule"`
	SelectedBlock int64                      `json:"selected_block"`
	Pool          string                     `json:"pool"`
	Declined      bool                       `json:"declined"`
	Candidates    []HotPrefixChoiceCandidate `json:"candidates"`
}

type HotPrefixDiagnostics struct {
	Schema            string                  `json:"schema"`
	MeasureCPU        bool                    `json:"cpu_measurement_enabled"`
	Coverage          string                  `json:"cpu_coverage"`
	Choices           int64                   `json:"choices"`
	CandidateVisits   int64                   `json:"candidate_visits"`
	HistoryNodes      int                     `json:"history_nodes"`
	ShadowRecords     int                     `json:"shadow_records"`
	ShadowHeapRecords int                     `json:"shadow_heap_records"`
	RequestRecords    int                     `json:"request_records"`
	CreditedPairs     int                     `json:"credited_request_block_pairs"`
	CPU               map[string]HotPrefixCPU `json:"cpu"`
	Records           []HotPrefixChoice       `json:"-"`
}

func (p *HotPrefixPolicy) cpuStart() time.Time {
	if p.diagnostics != nil && p.diagnostics.MeasureCPU {
		return time.Now()
	}
	return time.Time{}
}

func (p *HotPrefixPolicy) cpuEnd(method string, start time.Time) {
	if start.IsZero() {
		return
	}
	c := p.diagnostics.CPU[method]
	c.Calls++
	c.Nanoseconds += time.Since(start).Nanoseconds()
	p.diagnostics.CPU[method] = c
}

func (p *HotPrefixPolicy) recordChoice(c PeerReclaimContext, d PeerDecision) {
	if p.diagnostics == nil {
		return
	}
	s := p.diagnostics
	s.Choices++
	s.CandidateVisits += int64(len(c.Candidates))
	if !p.config.Diagnostics.TraceCandidates {
		return
	}
	r := HotPrefixChoice{Sequence: s.Choices, ScoreRule: p.config.HBMScore, CandidateRule: p.config.HBMCandidates,
		SelectedBlock: d.BlockID, Pool: d.Pool, Declined: d.Decline}
	if r.ScoreRule == "" {
		r.ScoreRule = "paper"
	}
	if r.CandidateRule == "" {
		r.CandidateRule = "leaf"
	}
	parents := p.nonLeaves(c.ResidentHashes)
	copies := map[string]int{}
	for _, h := range c.ResidentHashes {
		copies[h]++
	}
	for i, v := range c.Candidates {
		x := HotPrefixChoiceCandidate{BlockID: v.ID, Hash: v.Hash, LRUIndex: i,
			Eligible: p.eligible(v, parents, copies), HasResidentDescendant: parents[v.Hash], Copies: copies[v.Hash],
			ReadyCopies: append([]string(nil), v.ReadyCopies...), LengthTokens: p.blockTokens, Score: p.score(v.Hash)}
		if n := p.nodes[v.Hash]; n != nil {
			x.Frequency = n.frequency
			x.Clock = n.clock
			x.Depth = n.depth
		}
		r.Candidates = append(r.Candidates, x)
	}
	s.Records = append(s.Records, r)
}

func (s *PeerCache) annotateHotPrefixChoice(request string, deficit int64) {
	if s.hotprefix == nil {
		return
	}
	d := s.hotprefix.policy.diagnostics
	if d == nil || len(d.Records) == 0 {
		return
	}
	r := &d.Records[len(d.Records)-1]
	r.TimeUS = s.clock
	r.Request = request
	r.DeficitBlocks = deficit
}

func (s *PeerCache) HotPrefixDiagnostics() *HotPrefixDiagnostics {
	if s.hotprefix == nil || s.hotprefix.policy.diagnostics == nil {
		return nil
	}
	h := s.hotprefix
	d := h.policy.diagnostics
	d.HistoryNodes = len(h.policy.nodes)
	d.ShadowRecords = len(h.shadows)
	d.ShadowHeapRecords = len(h.expiry)
	d.RequestRecords = len(h.requests)
	d.CreditedPairs = 0
	for _, r := range h.requests {
		d.CreditedPairs += len(r.credited)
	}
	return d
}
