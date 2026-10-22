package policylab

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

type DecisionPolicyConfig struct {
	DecodeFillServiceCost   *DecodeFillServiceCostConfig   `json:"decode_fill_service_cost,omitempty"`
	PrefillDeferralProbe    *PrefillDeferralProbeConfig    `json:"prefill_deferral_probe,omitempty"`
	PrefillDeferrals        bool                           `json:"prefill_deferrals,omitempty"`
	PreemptionStorage       *PreemptionStorageConfig       `json:"preemption_storage,omitempty"`
	RequestSpill            bool                           `json:"request_spill,omitempty"`
	QueueServiceCost        *QueueServiceCostConfig        `json:"queue_service_cost,omitempty"`
	CapacityReservationView bool                           `json:"capacity_reservation_view,omitempty"`
	WaitPostStepCost        *WaitPostStepCostConfig        `json:"wait_post_step_cost,omitempty"`
	WaitDecisionCost        *WaitDecisionCostConfig        `json:"wait_decision_cost,omitempty"`
	WaitHostServiceCost     *WaitHostServiceCostConfig     `json:"wait_host_service_cost,omitempty"`
	WaitExecutionCost       *WaitExecutionCostConfig       `json:"wait_execution_cost,omitempty"`
	PrefillWaitProbe        *PrefillWaitProbeConfig        `json:"prefill_wait_probe,omitempty"`
	PrefillWaits            bool                           `json:"prefill_waits,omitempty"`
	HostServiceCost         *CapacityHostServiceCostConfig `json:"host_service_cost,omitempty"`
	ExecutionCost           *CapacityExecutionCostConfig   `json:"execution_cost,omitempty"`
	ExecutionWork           bool                           `json:"execution_work,omitempty"`
	CapacityPreemption      bool                           `json:"capacity_preemption,omitempty"`
	PrefillPreemption       bool                           `json:"prefill_preemption,omitempty"`
	PrefixSharing           bool                           `json:"prefix_sharing,omitempty"`
	ProducerPrefixes        bool                           `json:"producer_prefixes,omitempty"`
	ProducerWait            *ProducerWaitConfig            `json:"producer_wait,omitempty"`
	BalancedBatch           *BalancedBatchConfig           `json:"balanced_batch,omitempty"`
	BudgetRestore           *BudgetRestoreConfig           `json:"budget_restore,omitempty"`
	RestorePrefixBlocks     *int64                         `json:"restore_prefix_blocks,omitempty"`
	QueueOrder              string                         `json:"queue_order,omitempty"`
	PrefillTokenCap         int64                          `json:"prefill_token_cap,omitempty"`
	ExtraCost               *sim.LinearDecisionCost        `json:"extra_cost,omitempty"`
	RestoreServiceCost      *RestoreServiceCostConfig      `json:"restore_service_cost,omitempty"`
	TraceEvents             bool                           `json:"trace_events,omitempty"`
	ControlSteps            bool                           `json:"control_steps,omitempty"`
	AdmissionDelayUS        int64                          `json:"admission_delay_us,omitempty"`
}

// Optional baseline decoration; custom joint policies can emit per-victim
// choices directly without selecting this fixed mode.
type PreemptionStorageConfig struct {
	Guardband float64 `json:"guardband,omitempty"`
	Mode      string  `json:"mode"`
	Pool      string  `json:"pool,omitempty"`
}

func (c PreemptionStorageConfig) NewPolicy(base sim.DecisionPolicy) (sim.DecisionPolicy, error) {
	if c.Mode == "budget" {
		guard := c.Guardband
		if guard == 0 {
			guard = 1
		}
		return sim.NewBudgetPreemptionStoragePolicy(base, c.Pool, guard)
	}
	if c.Guardband != 0 {
		return nil, fmt.Errorf("spill guardband is only used by budget mode")
	}
	return sim.NewPreemptionStoragePolicy(base, c.Mode, c.Pool)
}

// PrefillWaitProbeConfig configures a one-shot resident-prefill mechanism probe.
// Omitted/null requests select all; an explicit empty list selects none.
type PrefillWaitProbeConfig struct {
	DurationUS        int64    `json:"duration_us"`
	Requests          []string `json:"requests"`
	MinComputedTokens int64    `json:"min_computed_tokens,omitempty"`
}

// UnmarshalJSON keeps the Python probe's integer/string input contract. Go's
// default decoder would otherwise turn null durations/elements into zero values.
func (c *PrefillWaitProbeConfig) UnmarshalJSON(data []byte) error {
	var raw struct {
		DurationUS        *int64          `json:"duration_us"`
		Requests          []*string       `json:"requests"`
		MinComputedTokens json.RawMessage `json:"min_computed_tokens"`
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&raw); err != nil {
		return err
	}
	if raw.DurationUS == nil || *raw.DurationUS < 0 {
		return fmt.Errorf("prefill_wait_probe requires an integer nonnegative duration_us")
	}
	var minComputed int64
	if len(raw.MinComputedTokens) > 0 {
		var value *int64
		if err := json.Unmarshal(raw.MinComputedTokens, &value); err != nil || value == nil || *value < 0 {
			return fmt.Errorf("prefill_wait_probe min_computed_tokens must be a nonnegative integer")
		}
		minComputed = *value
	}
	var requests []string
	if raw.Requests != nil {
		requests = make([]string, len(raw.Requests))
		for i, request := range raw.Requests {
			if request == nil {
				return fmt.Errorf("prefill_wait_probe requests[%d] must be a string", i)
			}
			requests[i] = *request
		}
	}
	c.DurationUS, c.Requests = *raw.DurationUS, requests
	c.MinComputedTokens = minComputed
	return nil
}

// NewPolicy decorates an existing policy through the same probe factory used
// by Run, so replay tools and actual execution share the mechanism.
func (c PrefillWaitProbeConfig) NewPolicy(base sim.DecisionPolicy) (*sim.PrefillWaitProbe, error) {
	return sim.NewPrefillWaitProbeWithOptions(base, c.DurationUS, c.Requests, sim.PrefillWaitProbeOptions{MinComputedTokens: c.MinComputedTokens})
}

type ProducerWaitConfig struct {
	MinSharedTokens int64 `json:"min_shared_tokens"`
	MaxWaitUS       int64 `json:"max_wait_us"`
}

type BalancedBatchConfig struct {
	DecodeFill          bool    `json:"decode_fill,omitempty"`
	ReadyComputeBudget  bool    `json:"ready_compute_budget,omitempty"`
	BundleHits          bool    `json:"bundle_hits,omitempty"`
	MaxLoadComputeRatio float64 `json:"max_load_compute_ratio"`
	TokenBudget         int64   `json:"token_budget,omitempty"`
	MaxRequests         int64   `json:"max_requests,omitempty"`
}

type PrefillDeferralProbeConfig struct {
	MaxRounds         int64    `json:"max_rounds"`
	MinComputedTokens int64    `json:"min_computed_tokens,omitempty"`
	Requests          []string `json:"requests"`
}

func (c PrefillDeferralProbeConfig) NewPolicy(base sim.DecisionPolicy) (sim.DecisionPolicy, error) {
	return sim.NewPrefillDeferralProbe(base, c.MaxRounds, c.MinComputedTokens, c.Requests)
}

func (c BalancedBatchConfig) newPolicy(cap int64) (sim.DecisionPolicy, error) {
	base, err := sim.NewBalancedBatchPolicyWithOptions(c.MaxLoadComputeRatio, c.TokenBudget, c.MaxRequests, cap,
		sim.BalancedBatchOptions{BundleHits: c.BundleHits, ReadyComputeBudget: c.ReadyComputeBudget})
	if err != nil {
		return nil, err
	}
	if c.DecodeFill {
		return sim.NewDecodeFillPolicy(base)
	}
	return base, nil
}

type BudgetRestoreConfig struct {
	ReestimateOnDecodeDrop bool    `json:"reestimate_on_decode_drop,omitempty"`
	PrimaryWeight          int64   `json:"primary_weight,omitempty"`
	BestEffortWeight       int64   `json:"best_effort_weight,omitempty"`
	Mode                   string  `json:"mode,omitempty"`
	Guardband              float64 `json:"guardband"`
}

func (c BudgetRestoreConfig) newPolicy(cap int64) (sim.DecisionPolicy, error) {
	if c.ReestimateOnDecodeDrop && c.Mode != "budget_queues" {
		return nil, fmt.Errorf("decode-drop budget reestimate requires budget_queues mode")
	}
	if c.Mode != "budget_queues" && (c.PrimaryWeight != 0 || c.BestEffortWeight != 0) {
		return nil, fmt.Errorf("queue share weights require budget_queues mode")
	}
	switch c.Mode {
	case "budget_queues":
		primary, best := c.PrimaryWeight, c.BestEffortWeight
		if primary == 0 && best == 0 {
			primary, best = 4, 1
		}
		return sim.NewBudgetQueuePolicyWithOptions(c.Guardband, cap, primary, best,
			sim.BudgetQueueOptions{ReestimateOnDecodeDrop: c.ReestimateOnDecodeDrop})
	case "", "restore_bound":
		return sim.NewBudgetRestorePolicy(c.Guardband, cap)
	case "completion_bound":
		return sim.NewCompletionBudgetRestorePolicy(c.Guardband, cap)
	case "initial_budget":
		return sim.NewInitialBudgetRestorePolicy(c.Guardband, cap)
	case "budget_pressure":
		return sim.NewBudgetPressurePolicy(c.Guardband, cap)
	case "budget_pressure_fcfs":
		return sim.NewBudgetPressureFIFOPolicy(c.Guardband, cap)
	default:
		return nil, fmt.Errorf("unknown budget restore mode %q", c.Mode)
	}
}

// decisionAdapter retains the existing queue observation at the same boundary
// for baseline comparisons. Metrics/resource operations stay in the runtime.
type decisionAdapter struct {
	policy   sim.DecisionPolicy
	order    string
	instance string
	sink     kv.PeerSink
}

func (a *decisionAdapter) Decide(v sim.DecisionView) sim.DecisionPlan {
	p := a.policy.Decide(v)
	if a.sink != nil && len(v.Waiting) > 1 {
		ids := append([]string(nil), p.QueueOrder...)
		if p.QueueOrder == nil {
			for _, r := range v.Waiting {
				ids = append(ids, r.ID)
			}
		}
		a.sink(kv.PeerRecord{Time: v.NowUS, Name: "request_queue_order", Instance: a.instance, Reason: a.order, Requests: ids})
	}
	return p
}
func (a *decisionAdapter) Observe(f sim.DecisionFeedback) { a.policy.Observe(f) }

type eventDecisionAdapter struct {
	*decisionAdapter
	listener sim.DecisionEventListener
}

func (a *eventDecisionAdapter) OnEvent(e sim.DecisionEvent) { a.listener.OnEvent(e) }
