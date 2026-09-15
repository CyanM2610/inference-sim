package policylab

import (
	"fmt"
	"github.com/inference-sim/inference-sim/sim"
	"math"
	"strings"
)

// Completion rates price [constant, returned grants, finishing requests]. A
// nil stage is disabled; an explicit zero preserves the prior execution times.
// RegistrationUS is additional controller work, not native add_request's base
// cost. Worker reset and unwrapped native-loop residuals are separate.
type CapacityHostServiceCostConfig struct {
	profileReporter
	RegistrationBaseUS   int64               `json:"registration_base_us,omitempty"`
	CompletionRatesUS    []int64             `json:"completion_rates_us,omitempty"`
	RegistrationUS       *int64              `json:"registration_us,omitempty"`
	MaxGrantedRequests   int64               `json:"max_granted_requests"`
	MaxFinishingRequests int64               `json:"max_finishing_requests"`
	MaxInputTokens       int64               `json:"max_input_tokens"`
	MaxOutputTokens      int                 `json:"max_output_tokens"`
	HBMBlocks            int64               `json:"hbm_blocks"`
	Shape                RestoreServiceShape `json:"shape"`
	Provenance           string              `json:"provenance"`
}

type capacityHostServiceCost struct{ config CapacityHostServiceCostConfig }

func newCapacityHostServiceCost(c Config) (*capacityHostServiceCost, error) {
	if c.DecisionPolicy == nil || c.DecisionPolicy.HostServiceCost == nil || !c.DecisionPolicy.CapacityPreemption || c.EnginePhases == nil || !c.DecisionPolicy.ControlSteps {
		return nil, fmt.Errorf("capacity host services require the capacity engine-phase path with control steps")
	}
	p := *c.DecisionPolicy.HostServiceCost
	p.profileReporter = c.profile("capacity_host", p.Provenance)
	if p.RegistrationBaseUS < 0 || p.RegistrationBaseUS > 0 && (p.RegistrationUS == nil || c.BatchCost == nil || c.BatchCost.EnqueueUS != 0) {
		return nil, fmt.Errorf("serialized registration base cost requires registration service and zero external enqueue cost")
	}
	if !validServiceShape(p.Shape) || p.MaxInputTokens <= 0 || p.MaxOutputTokens <= 0 || len(c.Instances) != 1 || p.HBMBlocks <= 0 || strings.TrimSpace(p.Provenance) == "" {
		return nil, fmt.Errorf("invalid host service geometry, input/output bounds or provenance")
	}
	p.equal("hbm_blocks", c.Instances[0].HBMBlocks, p.HBMBlocks)
	p.profileReporter.shape(p.Shape, serviceShape(c))
	if err := p.requests(c, p.MaxInputTokens, p.MaxOutputTokens); err != nil {
		return nil, err
	}
	if p.CompletionRatesUS == nil && p.RegistrationUS == nil {
		return nil, fmt.Errorf("host service profile has no enabled stage")
	}
	if p.MaxGrantedRequests < 0 || p.MaxFinishingRequests < 0 || p.RegistrationUS != nil && *p.RegistrationUS < 0 || p.CompletionRatesUS != nil && len(p.CompletionRatesUS) != 3 {
		return nil, fmt.Errorf("invalid host service coefficients or limits")
	}
	for _, rate := range p.CompletionRatesUS {
		if rate < 0 {
			return nil, fmt.Errorf("negative completion coefficient")
		}
	}
	p.CompletionRatesUS = append([]int64(nil), p.CompletionRatesUS...)
	if p.RegistrationUS != nil {
		rate := *p.RegistrationUS
		p.RegistrationUS = &rate
	}
	return &capacityHostServiceCost{config: p}, nil
}

func (m *capacityHostServiceCost) EstimateHostService(w sim.HostServiceWork) (sim.DecisionCostEstimate, error) {
	p := m.config
	var total int64
	switch w.Stage {
	case "registration":
		if p.RegistrationUS == nil || len(w.Requests) != 1 || len(w.Finishing) != 0 {
			return sim.DecisionCostEstimate{}, fmt.Errorf("unsupported registration work")
		}
		total = *p.RegistrationUS
		if p.RegistrationBaseUS > math.MaxInt64-total {
			return sim.DecisionCostEstimate{}, fmt.Errorf("registration cost overflow")
		}
		total += p.RegistrationBaseUS
	case "completion":
		if len(p.CompletionRatesUS) != 3 {
			return sim.DecisionCostEstimate{}, fmt.Errorf("missing completion cost formula")
		}
		p.above("granted_requests", int64(len(w.Requests)), p.MaxGrantedRequests)
		p.above("finishing_requests", int64(len(w.Finishing)), p.MaxFinishingRequests)
		requests := map[string]bool{}
		for _, id := range w.Requests {
			if requests[id] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("repeated completion request")
			}
			requests[id] = true
		}
		for _, id := range w.Finishing {
			if !requests[id] {
				return sim.DecisionCostEstimate{}, fmt.Errorf("finishing request absent or repeated")
			}
			delete(requests, id)
		}
		for i, n := range []int64{1, int64(len(w.Requests)), int64(len(w.Finishing))} {
			rate := p.CompletionRatesUS[i]
			if n > 0 && rate > (math.MaxInt64-total)/n {
				return sim.DecisionCostEstimate{}, fmt.Errorf("completion cost overflow")
			}
			total += rate * n
		}
	default:
		return sim.DecisionCostEstimate{}, fmt.Errorf("unknown host service stage %q", w.Stage)
	}
	return sim.DecisionCostEstimate{ExtraUS: total, Provenance: p.Provenance}, nil
}
