package kvruntime

import "testing"

func validPagedProfilePoint() PagedForwardPoint {
	p := PagedForwardPoint{Prefix: 16, Query: 1, BeforeAllocatedBytes: 15 << 30, ReservedBytes: 16 << 30, Evidence: "test measured shape", Stages: map[string]BridgeStage{}, PhasePeakDelta: map[string]int64{}, PhaseEndDelta: map[string]int64{}}
	for _, name := range []string{"gather_prefix", "forward", "scatter_delta", "hf_release"} {
		p.Stages[name] = BridgeStage{WallNS: 1000, SubmitNS: 1, ServiceNS: 1}
		p.PhasePeakDelta[name] = 1 << 20
		p.PhaseEndDelta[name] = 0
	}
	return p
}

func TestPagedProfileRejectsMissingCostsAndImpossibleMemory(t *testing.T) {
	if err := validPagedProfilePoint().Validate(); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*PagedForwardPoint){
		"missing phase":         func(p *PagedForwardPoint) { delete(p.Stages, "gather_prefix") },
		"missing peak":          func(p *PagedForwardPoint) { delete(p.PhasePeakDelta, "forward") },
		"zero mapping cost":     func(p *PagedForwardPoint) { p.Stages["scatter_delta"] = BridgeStage{WallNS: 1} },
		"missing evidence":      func(p *PagedForwardPoint) { p.Evidence = "" },
		"arena overflow":        func(p *PagedForwardPoint) { p.Query = 625 },
		"reserved below active": func(p *PagedForwardPoint) { p.ReservedBytes = p.BeforeAllocatedBytes - 1 },
		"peak beyond reserved":  func(p *PagedForwardPoint) { p.PhasePeakDelta["forward"] = 2 << 30 },
		"end above peak":        func(p *PagedForwardPoint) { p.PhaseEndDelta["forward"] = 2 << 20 },
	} {
		t.Run(name, func(t *testing.T) {
			p := validPagedProfilePoint()
			change(&p)
			if p.Validate() == nil {
				t.Fatal("invalid calibration accepted")
			}
		})
	}
	p := validPagedProfilePoint()
	profile := Profile{Version: 1, QueueDepth: 1, SourceSHA256: map[string]string{"test": "hash"}, PagedForward: []PagedForwardPoint{p, p}}
	if profile.Validate() == nil {
		t.Fatal("duplicate paged shape accepted")
	}
}
