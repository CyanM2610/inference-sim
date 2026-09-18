// Package policylab composes the real BLIS cluster engine with a peer-L2 KVStore.
// It owns experiment I/O, not a second simulator or a replacement compute model.
package policylab

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
	"github.com/inference-sim/inference-sim/sim/latency"
)

type InstanceConfig struct {
	HBMBlocks int64           `json:"hbm_blocks"`
	Access    []kv.PeerAccess `json:"access"`
}
type PolicyConfig struct {
	Name         string `json:"name"`
	MinFrequency int64  `json:"min_frequency"`
	MaxStoreUS   int64  `json:"max_store_us"`
}
type RequestConfig struct {
	AfterRequest    string        `json:"after_request,omitempty"`
	ThinkTimeUS     int64         `json:"think_time_us,omitempty"`
	ID              string        `json:"id"`
	At              int64         `json:"at_us"`
	Input           []sim.TokenID `json:"input_tokens"`
	Output          []sim.TokenID `json:"output_tokens"`
	TTFTSLOUS       int64         `json:"ttft_slo_us,omitempty"`
	MaxOutputTokens int           `json:"max_output_tokens,omitempty"`
}
type PromotionConfig struct {
	Instance int           `json:"instance"`
	At       int64         `json:"at_us"`
	Tokens   []sim.TokenID `json:"prefix_tokens"`
	Budget   int           `json:"block_budget"`
}
type Config struct {
	profileWarnings           *profileWarnings
	HotPrefix                 *kv.HotPrefixConfig        `json:"hotprefix,omitempty"`
	DirectionalTransferOrder  bool                       `json:"directional_transfer_order,omitempty"`
	TransferSubmissionPolicy  string                     `json:"transfer_submission_policy,omitempty"`
	TransferSubmissionCost    *kv.TransferSubmissionCost `json:"transfer_submission_cost,omitempty"`
	PrefixCopySelection       string                     `json:"prefix_copy_selection,omitempty"`
	DecodeCapacityReservation bool                       `json:"decode_capacity_reservation,omitempty"`
	PromotionRetention        string                     `json:"promotion_retention,omitempty"`
	PromotionControl          bool                       `json:"promotion_control,omitempty"`
	BatchPrefixReuse          bool                       `json:"batch_prefix_reuse,omitempty"`
	DecisionEstimates         *DecisionEstimateConfig    `json:"decision_estimates,omitempty"`
	RestoreControl            bool                       `json:"restore_control,omitempty"`
	DecisionPolicy            *DecisionPolicyConfig      `json:"decision_policy,omitempty"`
	EnginePhases              *EnginePhaseCostConfig     `json:"engine_phases,omitempty"`
	StoreSourceReuse          bool                       `json:"store_source_reuse,omitempty"`
	TraceBatchShapes          bool                       `json:"trace_batch_shapes,omitempty"`
	BatchCost                 *BatchCostConfig           `json:"batch_cost,omitempty"`
	Native                    *NativeConfig              `json:"native_runtime,omitempty"`
	Seed                      int64                      `json:"seed"`
	RequestScheduler          string                     `json:"request_scheduler,omitempty"`
	BlockTokens               int64                      `json:"block_tokens"`
	MaxBatchTokens            int64                      `json:"max_batch_tokens"`
	MaxSequences              int64                      `json:"max_sequences"`
	PrefillChunk              int64                      `json:"prefill_chunk"`
	Model                     sim.ModelConfig            `json:"model"`
	Hardware                  sim.HardwareCalib          `json:"hardware"`
	Alpha                     []float64                  `json:"alpha_us"`
	Routing                   string                     `json:"routing"`
	RoutingUS                 int64                      `json:"routing_us"`
	Resources                 []kv.PeerResource          `json:"resources"`
	Pools                     []kv.PeerPoolConfig        `json:"pools"`
	Instances                 []InstanceConfig           `json:"instances"`
	Policy                    PolicyConfig               `json:"policy"`
	Requests                  []RequestConfig            `json:"requests"`
	Promotions                []PromotionConfig          `json:"promotions"`
	Mechanisms                *kv.PeerMechanisms         `json:"mechanisms,omitempty"`
}
type NativeConfig struct {
	InputCopies      bool   `json:"input_copies,omitempty"`
	OutputCopies     bool   `json:"output_copies,omitempty"`
	CopyEnginePolicy string `json:"copy_engine_policy,omitempty"`
	TorchCopies      bool   `json:"torch_copies,omitempty"`
	Profile          string `json:"profile"`
	NUMA             int    `json:"numa"`
	ComputeGrid      bool   `json:"compute_grid,omitempty"`
	PagedForward     bool   `json:"paged_forward,omitempty"`
}
type CPUProfile struct {
	Count       int64 `json:"count"`
	Nanoseconds int64 `json:"wall_ns"`
}
type Result struct {
	EnginePhaseObservations             []kv.EnginePhaseObservation   `json:"-"`
	CacheMetrics                        CacheMetrics                  `json:"cache_metrics"`
	ArrivalUS                           map[string]int64              `json:"arrival_us,omitempty"`
	ProfileWarnings                     []ProfileWarning              `json:"profile_warnings,omitempty"`
	ProfileCostCoverage                 string                        `json:"profile_cost_coverage,omitempty"`
	VLLMNativeRevision                  string                        `json:"vllm_native_revision,omitempty"`
	VLLMNativeCostCoverage              string                        `json:"vllm_native_cost_coverage,omitempty"`
	HotPrefixCostCoverage               string                        `json:"hotprefix_cost_coverage,omitempty"`
	DirectionalTransferOrderCoverage    string                        `json:"directional_transfer_order_coverage,omitempty"`
	TransferSubmissions                 []kv.TransferSubmissionRecord `json:"-"`
	TransferSubmissionCostCoverage      string                        `json:"transfer_submission_cost_coverage,omitempty"`
	DecodeFillServiceCostCoverage       string                        `json:"decode_fill_service_cost_coverage,omitempty"`
	SpillEstimateCoverage               string                        `json:"spill_estimate_coverage,omitempty"`
	RequestSpillCostCoverage            string                        `json:"request_spill_cost_coverage,omitempty"`
	QueueServiceCostCoverage            string                        `json:"queue_service_cost_coverage,omitempty"`
	WorkerMetadataCostCoverage          string                        `json:"worker_metadata_cost_coverage,omitempty"`
	PostStepServices                    []sim.PostStepRecord          `json:"-"`
	PostStepCostCoverage                string                        `json:"post_step_cost_coverage,omitempty"`
	WaitExecutionCostCoverage           string                        `json:"wait_execution_cost_coverage,omitempty"`
	RequestRecomputeCostCoverage        string                        `json:"request_recompute_cost_coverage,omitempty"`
	HostServices                        []sim.HostServiceRecord       `json:"-"`
	HostServiceCostCoverage             string                        `json:"host_service_cost_coverage,omitempty"`
	PrefixCopyCostCoverage              string                        `json:"prefix_copy_cost_coverage,omitempty"`
	DecodeCapacityCostCoverage          string                        `json:"decode_capacity_cost_coverage,omitempty"`
	CapacityReservationViewCostCoverage string                        `json:"capacity_reservation_view_cost_coverage,omitempty"`
	DecisionAdditionalCostCoverage      string                        `json:"decision_additional_cost_coverage,omitempty"`
	PrefillPreemptionCostCoverage       string                        `json:"prefill_preemption_cost_coverage,omitempty"`
	PrefillWaitCostCoverage             string                        `json:"prefill_wait_cost_coverage,omitempty"`
	PrefillDeferralCostCoverage         string                        `json:"prefill_deferral_cost_coverage,omitempty"`
	PromotionCostCoverage               string                        `json:"promotion_cost_coverage,omitempty"`
	InjectedPolicies                    map[string]map[string]string  `json:"injected_policies,omitempty"`
	KVPolicyCostCoverage                string                        `json:"kv_policy_cost_coverage,omitempty"`
	BatchPrefixCostCoverage             string                        `json:"batch_prefix_cost_coverage,omitempty"`
	DecisionEstimateCoverage            string                        `json:"decision_estimate_coverage,omitempty"`
	RestoreCostCoverage                 string                        `json:"restore_cost_coverage,omitempty"`
	PolicyDecisions                     []sim.DecisionRecord          `json:"-"`
	DecisionCostCoverage                string                        `json:"decision_cost_coverage,omitempty"`
	PolicyEvents                        []sim.DecisionEventRecord     `json:"-"`
	DecisionEventCostCoverage           string                        `json:"decision_event_cost_coverage,omitempty"`
	BatchShapes                         []BatchCostObservation        `json:"batch_shapes,omitempty"`
	// Absolute first observation, captured once before any possible re-prefill
	// overwrites legacy core TTFT fields. These are the ranking API timestamps.
	FirstTokenUS      map[string]int64            `json:"first_token_us"`
	FinishedUS        map[string]int64            `json:"finished_us"`
	NativeAllocations map[string]map[string]int64 `json:"native_allocations,omitempty"`
	NativeEvents      []kvruntime.Record          `json:"-"`
	NativeProfile     *kvruntime.Profile          `json:"native_profile,omitempty"`
	NativeValidated   bool                        `json:"native_validated,omitempty"`
	Config            Config                      `json:"config"`
	BlockBytes        int64                       `json:"block_bytes"`
	Metrics           sim.MetricsOutput           `json:"metrics"`
	Requests          []sim.RequestMetrics        `json:"requests"`
	Events            []kv.PeerRecord             `json:"-"`
	Pools             map[string]map[string]int64 `json:"pools"`
	HBM               map[string]map[string]int64 `json:"hbm"`
	Counts            map[string]int64            `json:"counts"`
	CPU               map[string]CPUProfile       `json:"-"`
	RoutingTrace      any                         `json:"-"`
}

func (c Config) Validate() error {
	if err := c.validateConversationArrivals(); err != nil {
		return err
	}
	if c.RequestScheduler == "vllm_native" {
		if err := c.validateVLLMNative(); err != nil {
			return err
		}
	}
	if c.PrefixCopySelection != "" && c.PrefixCopySelection != "first_published" {
		return fmt.Errorf("prefix_copy_selection must be omitted or first_published")
	}
	if c.DecodeCapacityReservation && (c.PromotionControl || len(c.Promotions) > 0 || c.BatchPrefixReuse) {
		return fmt.Errorf("decode capacity reservations do not yet cover active promotions or batch prefix donors")
	}
	if c.PromotionRetention != "" && (!c.PromotionControl || (c.PromotionRetention != "cache" && c.PromotionRetention != "request")) {
		return fmt.Errorf("promotion_retention requires promotion_control and cache or request")
	}
	if c.PromotionRetention == "request" && !c.RestoreControl {
		return fmt.Errorf("request-retained promotion requires restore_control")
	}
	if c.PromotionControl && (c.DecisionPolicy == nil || c.EnginePhases == nil || c.Mechanisms == nil || !c.Mechanisms.GroupTransfers || !c.Mechanisms.ConcurrentRestores || len(c.Pools) == 0) {
		return fmt.Errorf("promotion_control requires decision_policy and grouped single-instance offload engine phases")
	}
	if c.DecisionEstimates != nil {
		if c.DecisionPolicy == nil {
			return fmt.Errorf("decision estimates require decision_policy")
		}
		if _, err := newRestoreEstimator(c); err != nil {
			return err
		}
	}
	if c.RestoreControl && (c.EnginePhases == nil || c.Mechanisms == nil || !c.Mechanisms.ConcurrentRestores || !c.Mechanisms.GroupTransfers) {
		return fmt.Errorf("restore_control requires engine_phases with grouped concurrent restores")
	}
	if c.BatchPrefixReuse && c.EnginePhases == nil {
		return fmt.Errorf("batch_prefix_reuse requires the single-instance engine-phase backend")
	}
	if c.DecisionPolicy != nil {
		if c.DecisionPolicy.DecodeFillServiceCost != nil {
			if _, err := newDecodeFillServiceCost(c); err != nil {
				return err
			}
		}
		if probe := c.DecisionPolicy.PrefillDeferralProbe; probe != nil {
			if !c.DecisionPolicy.PrefillDeferrals {
				return fmt.Errorf("prefill_deferral_probe requires prefill_deferrals")
			}
			if _, err := probe.NewPolicy(&sim.PrefillSJFPolicy{}); err != nil {
				return err
			}
		}
		if p := c.DecisionPolicy; p.PrefillDeferrals && (p.QueueServiceCost != nil || p.HostServiceCost != nil || p.ExecutionCost != nil ||
			p.RestoreServiceCost != nil || p.WaitExecutionCost != nil || p.WaitHostServiceCost != nil || p.WaitDecisionCost != nil || p.WaitPostStepCost != nil) {
			return fmt.Errorf("resident prefill deferrals are outside existing measured decision/host/execution service profiles")
		}
		if c.DecisionPolicy.QueueServiceCost != nil {
			if _, err := newQueueServiceCost(c); err != nil {
				return err
			}
		}
		if probe := c.DecisionPolicy.PrefillWaitProbe; probe != nil {
			if !c.DecisionPolicy.PrefillWaits {
				return fmt.Errorf("prefill_wait_probe requires prefill_waits")
			}
			if probe.DurationUS < 0 || probe.MinComputedTokens < 0 {
				return fmt.Errorf("prefill_wait_probe requires nonnegative duration_us and min_computed_tokens")
			}
		}
		if c.DecisionPolicy.PrefillWaits && (c.DecisionPolicy.HostServiceCost != nil || c.DecisionPolicy.ExecutionCost != nil || c.DecisionPolicy.RestoreServiceCost != nil) {
			return fmt.Errorf("resident prefill waits are outside existing host, execution and restore service profiles")
		}
		if c.DecisionPolicy.HostServiceCost != nil {
			if _, err := newCapacityHostServiceCost(c); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.ExecutionCost != nil {
			if _, err := newCapacityExecutionCost(c); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.WaitExecutionCost != nil {
			if _, err := newWaitExecutionCost(c); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.WaitHostServiceCost != nil {
			if _, err := newWaitHostServiceCost(c); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.WaitDecisionCost != nil {
			if _, err := newWaitDecisionCost(c); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.WaitPostStepCost != nil {
			if _, err := newWaitPostStepCost(c); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.PrefixSharing && c.EnginePhases == nil {
			return fmt.Errorf("prefix_sharing requires the single-instance engine-phase backend")
		}
		if w := c.DecisionPolicy.ProducerWait; w != nil {
			if !c.DecisionPolicy.ProducerPrefixes || c.DecisionPolicy.BalancedBatch == nil || w.MinSharedTokens <= 0 || w.MaxWaitUS <= 0 {
				return fmt.Errorf("producer_wait requires producer_prefixes, balanced_batch and positive token/age limits")
			}
		}
		if c.DecisionPolicy.ProducerPrefixes && c.EnginePhases == nil {
			return fmt.Errorf("producer_prefixes currently requires the engine-phase ready-prefix backend")
		}
		if balanced := c.DecisionPolicy.BalancedBatch; balanced != nil {
			if balanced.DecodeFill && (!c.DecisionPolicy.PrefillDeferrals || !c.PromotionControl || c.PromotionRetention != "request" ||
				c.DecisionPolicy.ProducerWait != nil || c.DecisionPolicy.PrefillWaitProbe != nil || c.DecisionPolicy.PrefillDeferralProbe != nil) {
				return fmt.Errorf("decode_fill requires round deferrals and request-retained promotion without another wait/deferral policy")
			}
			if balanced.BundleHits && !c.DecisionPolicy.PrefixSharing {
				return fmt.Errorf("bundle_hits requires prefix_sharing")
			}
			if balanced.TokenBudget > 0 && balanced.TokenBudget < c.MaxSequences {
				return fmt.Errorf("balanced token budget must cover one token per running sequence")
			}
			if c.DecisionPolicy.BudgetRestore != nil || c.DecisionPolicy.QueueOrder != "" || c.DecisionPolicy.RestorePrefixBlocks != nil || c.DecisionPolicy.AdmissionDelayUS != 0 || c.DecisionPolicy.RestoreServiceCost != nil || !c.RestoreControl {
				return fmt.Errorf("balanced_batch owns admission/queue/restore choices and requires its own cost coverage")
			}
			if _, err := balanced.newPolicy(c.DecisionPolicy.PrefillTokenCap); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.ControlSteps && c.EnginePhases == nil {
			return fmt.Errorf("control_steps requires engine_phases")
		}
		if budget := c.DecisionPolicy.BudgetRestore; budget != nil {
			if (budget.Mode == "budget_pressure" || budget.Mode == "budget_pressure_fcfs" || budget.Mode == "budget_queues") && !c.DecisionPolicy.CapacityPreemption {
				return fmt.Errorf("budget_pressure requires capacity_preemption")
			}
			if c.DecisionEstimates == nil || c.DecisionPolicy.QueueOrder != "" || c.DecisionPolicy.RestorePrefixBlocks != nil || c.DecisionPolicy.AdmissionDelayUS != 0 {
				return fmt.Errorf("budget_restore requires estimates and owns queue/restore selection")
			}
			if _, err := budget.newPolicy(c.DecisionPolicy.PrefillTokenCap); err != nil {
				return err
			}
		}
		if cap := c.DecisionPolicy.RestorePrefixBlocks; cap != nil && (*cap < 0 || !c.RestoreControl) {
			return fmt.Errorf("restore_prefix_blocks requires restore_control and a nonnegative limit")
		}
		if c.DecisionPolicy.CapacityPreemption && (!c.DecisionPolicy.PrefillPreemption || !c.DecodeCapacityReservation || c.BatchPrefixReuse) {
			return fmt.Errorf("capacity_preemption requires prefill_preemption and decode_capacity_reservation without batch prefix reuse")
		}
		if c.DecisionPolicy.CapacityReservationView && !c.DecisionPolicy.CapacityPreemption {
			return fmt.Errorf("capacity_reservation_view requires capacity_preemption")
		}
		if c.DecisionPolicy.RequestSpill {
			if !c.DecisionPolicy.PrefillPreemption || c.Native != nil || c.Mechanisms != nil && (c.Mechanisms.ReserveAtDispatch || c.Mechanisms.Online != nil) {
				return fmt.Errorf("request_spill requires abstract/phase prefill preemption with immediate reservations")
			}
			if c.DecisionPolicy.QueueServiceCost != nil && c.DecisionPolicy.QueueServiceCost.Spill == nil || c.DecisionPolicy.ExecutionCost != nil || c.DecisionPolicy.HostServiceCost != nil || c.DecisionPolicy.WaitExecutionCost != nil || c.DecisionPolicy.WaitDecisionCost != nil || c.DecisionPolicy.WaitHostServiceCost != nil || c.DecisionPolicy.WaitPostStepCost != nil {
				return fmt.Errorf("request_spill is not covered by existing calibrated CPU service profiles")
			}
			for _, instance := range c.Instances {
				if len(instance.Access) == 0 {
					return fmt.Errorf("request_spill requires an accessible pool per instance")
				}
			}
		}
		if storage := c.DecisionPolicy.PreemptionStorage; storage != nil {
			if !c.DecisionPolicy.RequestSpill {
				return fmt.Errorf("preemption_storage requires request_spill")
			}
			queue, _ := sim.NewQueueDecisionPolicy("fcfs", 0)
			var base sim.DecisionPolicy = queue
			if storage.Mode == "budget" {
				budget := c.DecisionPolicy.BudgetRestore
				if c.DecisionEstimates == nil || !c.DecisionEstimates.Spill || budget == nil {
					return fmt.Errorf("budget spill requires spill estimates and retained-budget policy")
				}
				switch budget.Mode {
				case "initial_budget", "budget_pressure", "budget_pressure_fcfs", "budget_queues":
				default:
					return fmt.Errorf("budget spill requires a retained-budget policy")
				}
				var err error
				base, err = budget.newPolicy(c.DecisionPolicy.PrefillTokenCap)
				if err != nil {
					return err
				}
			}
			if _, err := storage.NewPolicy(base); err != nil {
				return err
			}
			if storage.Mode == "spill" || storage.Mode == "budget" {
				for _, instance := range c.Instances {
					found := false
					for _, access := range instance.Access {
						found = found || access.Pool == storage.Pool
					}
					if !found {
						return fmt.Errorf("preemption storage target is not accessible on every instance")
					}
				}
			}
		}
		if c.DecisionPolicy.AdmissionDelayUS < 0 {
			return fmt.Errorf("negative admission delay")
		}
		for _, request := range c.Requests {
			if request.At > math.MaxInt64-c.DecisionPolicy.AdmissionDelayUS {
				return fmt.Errorf("admission delay deadline overflows for %q", request.ID)
			}
		}
		if c.DecisionPolicy.ExtraCost != nil {
			if err := c.DecisionPolicy.ExtraCost.Validate(); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.RestoreServiceCost != nil {
			if c.DecisionPolicy.ExtraCost != nil {
				return fmt.Errorf("choose one decision service cost model")
			}
			if _, err := newRestoreServiceCost(c); err != nil {
				return err
			}
		}
		if c.DecisionPolicy.PrefillTokenCap > c.MaxBatchTokens {
			return fmt.Errorf("decision prefill cap exceeds batch token limit")
		}
		order := c.DecisionPolicy.QueueOrder
		if order == "" {
			order = c.decisionQueueOrder()
		}
		if _, err := sim.NewQueueDecisionPolicy(order, c.DecisionPolicy.PrefillTokenCap); err != nil {
			return err
		}
	}
	if c.TransferSubmissionPolicy != "" {
		if _, err := kv.NewTransferSubmissionPolicy(c.TransferSubmissionPolicy); err != nil {
			return err
		}
	}
	if c.DirectionalTransferOrder && c.EnginePhases == nil {
		return fmt.Errorf("directional_transfer_order requires engine_phases")
	}
	if c.TransferSubmissionPolicy != "" || c.TransferSubmissionCost != nil {
		if c.EnginePhases == nil {
			return fmt.Errorf("transfer submission policy/cost requires engine_phases")
		}
		if err := c.TransferSubmissionCost.Validate(); err != nil {
			return err
		}
	}
	if c.StoreSourceReuse && c.EnginePhases == nil {
		return fmt.Errorf("store_source_reuse requires engine_phases")
	}
	if c.EnginePhases != nil {
		if c.BatchCost == nil || c.Native != nil || len(c.Instances) != 1 || len(c.Promotions) != 0 {
			return fmt.Errorf("engine phases require a single batch-cost instance without native runtime or promotions")
		}
		if err := c.EnginePhases.Validate(); err != nil {
			return err
		}
	}
	if c.TraceBatchShapes && c.BatchCost == nil {
		return fmt.Errorf("trace_batch_shapes requires the batch_cost backend")
	}
	if c.BatchCost != nil {
		if c.Native != nil {
			return fmt.Errorf("batch_cost and native_runtime are separate execution backends")
		}
		if err := c.BatchCost.Validate(); err != nil {
			return err
		}
	}
	if c.Native != nil {
		if c.Native.InputCopies && !c.Native.OutputCopies {
			return fmt.Errorf("input copies require output_copies")
		}
		if c.Native.OutputCopies && (!c.Native.TorchCopies || !c.Native.PagedForward) {
			return fmt.Errorf("output copies require paged_forward and torch_copies")
		}
		if c.Native.CopyEnginePolicy != "" && (!c.Native.OutputCopies || (c.Native.CopyEnginePolicy != "fifo" && c.Native.CopyEnginePolicy != "stream_drain")) {
			return fmt.Errorf("copy_engine_policy requires output_copies and fifo/stream_drain")
		}
	}
	if c.Native != nil && c.Native.TorchCopies && (!c.Native.PagedForward || len(c.Pools) != 1 || c.Pools[0].CapacityBlocks != 64) {
		return fmt.Errorf("live Torch copy calibration requires paged_forward and the measured 64-slot DRAM pool")
	}
	if c.Native != nil && c.Native.PagedForward && (!c.Native.ComputeGrid || len(c.Instances) != 1 || c.Instances[0].HBMBlocks != 40 || c.Native.NUMA != 0) {
		return fmt.Errorf("paged forward requires compute_grid, one 40-block arena and measured NUMA 0")
	}
	if c.Native != nil && c.Native.ComputeGrid && (c.MaxSequences != 1 || c.PrefillChunk != 128 || c.MaxBatchTokens < 128) {
		return fmt.Errorf("native compute grid requires batch one and a 128-token prefill chunk")
	}
	if c.Native != nil && (len(c.Instances) != 1 || c.Native.Profile == "" || c.Model.NumLayers != 28 || c.Model.NumKVHeads != 4 || c.Model.EffectiveHeadDim() != 128 || c.Model.EffectiveKVBytesPerParam() != 2) {
		return fmt.Errorf("native profiling supports one Qwen2.5-7B BF16 GQA instance")
	}
	switch c.RequestScheduler {
	case "", "fcfs", "sjf", "edf", "vllm_native":
	default:
		return fmt.Errorf("request_scheduler must be fcfs, sjf, edf or vllm_native")
	}
	if c.Mechanisms != nil {
		if c.Mechanisms.BackgroundStoreMode == "on_preemption" && (c.DecisionPolicy == nil || !c.DecisionPolicy.RequestSpill) {
			return fmt.Errorf("on_preemption storage requires request_spill decisions")
		}
		if c.Mechanisms.GroupTransfers && c.Native != nil {
			return fmt.Errorf("grouped abstract transfers cannot use the detailed native runtime")
		}
		if err := c.Mechanisms.Validate(); err != nil {
			return err
		}
	}
	if len(c.Instances) == 0 || len(c.Requests) == 0 || c.BlockTokens <= 0 || c.MaxSequences <= 0 || c.MaxBatchTokens <= 0 || c.PrefillChunk < 0 || c.RoutingUS < 0 {
		return fmt.Errorf("positive instances/block/batch sizes and nonnegative timings required")
	}
	if c.Routing != "" && c.Routing != "round-robin" && c.Routing != "least-loaded" {
		return fmt.Errorf("runner supports round-robin or least-loaded routing")
	}
	switch c.Policy.Name {
	case "lru_drop", "lfu_store", "ready_first", "cost_aware", "hotprefix":
	default:
		return fmt.Errorf("unknown peer policy %q", c.Policy.Name)
	}
	if c.Policy.MinFrequency < 0 || c.Policy.MaxStoreUS < 0 {
		return fmt.Errorf("policy thresholds must be nonnegative")
	}
	if c.Policy.Name == "hotprefix" || c.HotPrefix != nil {
		if err := c.validateHotPrefix(); err != nil {
			return err
		}
	}
	if c.Model.NumLocalExperts != 0 || c.Model.KVLoraRank != 0 || c.Model.KVBearingLayers != 0 {
		return fmt.Errorf("this runner's calibrated scope is homogeneous dense MHA/GQA, TP=1")
	}
	hw := sim.NewModelHardwareConfig(c.Model, c.Hardware, "policy-model", "configured-gpu", 1, 1, false, "", "roofline", 0)
	if _, err := latency.NewLatencyModel(sim.NewLatencyCoeffs(nil, c.Alpha), hw); err != nil {
		return err
	}
	ids := map[string]bool{}
	minHBM := int64(math.MaxInt64)
	for _, i := range c.Instances {
		if i.HBMBlocks <= 0 {
			return fmt.Errorf("HBM capacity must be positive")
		}
		minHBM = min(minHBM, i.HBMBlocks)
	}
	for _, r := range c.Requests {
		if r.ID == "" || ids[r.ID] || r.At < 0 || len(r.Input) == 0 || r.TTFTSLOUS < 0 || r.MaxOutputTokens < 0 {
			return fmt.Errorf("invalid/duplicate request %q", r.ID)
		}
		if r.TTFTSLOUS > math.MaxInt64-r.At {
			return fmt.Errorf("request %s deadline overflows int64", r.ID)
		}
		if r.MaxOutputTokens > 0 && len(r.Output) > r.MaxOutputTokens {
			return fmt.Errorf("request %s exceeds its declared output budget", r.ID)
		}
		if c.DecodeCapacityReservation {
			if r.MaxOutputTokens <= 0 || int64(len(r.Input)) > math.MaxInt64-int64(r.MaxOutputTokens)+1 {
				return fmt.Errorf("decode capacity reservation needs an explicit finite output budget for %s", r.ID)
			}
			peak := int64(len(r.Input)) + int64(r.MaxOutputTokens) - 1
			if (peak-1)/c.BlockTokens+1 > minHBM {
				return fmt.Errorf("declared output capacity for %s exceeds HBM", r.ID)
			}
		}
		ids[r.ID] = true
		kvTokens := int64(len(r.Input) + max(len(r.Output)-1, 0))
		if (kvTokens+c.BlockTokens-1)/c.BlockTokens > minHBM {
			return fmt.Errorf("request %s exceeds the smallest instance HBM; reduce context or increase hbm_blocks", r.ID)
		}
	}
	for _, p := range c.Promotions {
		if p.Instance < 0 || p.Instance >= len(c.Instances) || p.At < 0 || p.Budget <= 0 || len(p.Tokens) == 0 {
			return fmt.Errorf("invalid promotion")
		}
	}
	return nil
}
func Run(c Config) (*Result, error) {
	return RunWithPolicies(c, PolicyFactories{})
}

// RunWithDecisionPolicy lets experiments inject a fresh stateful policy per
// instance without changing resource management or event/metric code. The
// decision_policy config must be enabled; its built-in ordering/cap settings
// are replaced by the factory's implementation.
func RunWithDecisionPolicy(c Config, factory func(instance string) sim.DecisionPolicy) (*Result, error) {
	if c.DecisionPolicy == nil || factory == nil {
		return nil, fmt.Errorf("custom decision factory requires decision_policy and a nonnil factory")
	}
	return RunWithPolicies(c, PolicyFactories{Decision: factory})
}

func run(c Config, factories PolicyFactories) (*Result, error) {
	warnings := &profileWarnings{}
	c.profileWarnings = warnings
	if c.EnginePhases != nil {
		phase := *c.EnginePhases
		phase.profileReporter = c.profile("engine_phases", phase.Provenance)
		c.EnginePhases = &phase
	}
	if c.RequestScheduler == "vllm_native" && factories.Decision != nil {
		return nil, fmt.Errorf("vllm_native cannot combine with an injected request controller")
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	hasSubmissionPolicy := factories.TransferSubmission != nil || c.TransferSubmissionPolicy != ""
	if c.HotPrefix != nil && (factories.Peer != nil || factories.PoolEviction != nil || hasSubmissionPolicy) {
		return nil, fmt.Errorf("HotPrefix owns coupled KV/pool placement; conflicting placement/transfer overrides are unsupported")
	}
	if hasSubmissionPolicy && c.EnginePhases == nil {
		return nil, fmt.Errorf("transfer submission factory requires engine_phases")
	}
	if c.TransferSubmissionCost != nil && !hasSubmissionPolicy {
		return nil, fmt.Errorf("transfer submission cost requires a policy")
	}
	if c.DecisionPolicy != nil && c.DecisionPolicy.DecodeFillServiceCost != nil && (factories.Decision != nil || factories.Peer != nil || factories.PoolEviction != nil || hasSubmissionPolicy) {
		return nil, fmt.Errorf("decode-fill service does not calibrate injected policy callbacks")
	}
	if c.DecisionPolicy != nil && c.DecisionPolicy.QueueServiceCost != nil && (factories.Decision != nil || factories.Peer != nil || factories.PoolEviction != nil || hasSubmissionPolicy) {
		return nil, fmt.Errorf("queue service costs do not calibrate injected policy callbacks")
	}
	bytes, err := latency.KVBytesPerToken(c.Model, 1)
	if err != nil {
		return nil, err
	}
	out := &Result{Config: c, BlockBytes: int64(math.Ceil(bytes * float64(c.BlockTokens))), Counts: map[string]int64{}, CPU: map[string]CPUProfile{}, HBM: map[string]map[string]int64{}}
	defer func() {
		out.ProfileWarnings = warnings.snapshot()
		if len(out.ProfileWarnings) > 0 {
			out.ProfileCostCoverage = "extrapolated_outside_measured_envelopes; see profile_warnings; no new native validation"
		}
	}()
	if c.RequestScheduler == "vllm_native" {
		out.VLLMNativeRevision = sim.VLLMNativeRevision
		out.VLLMNativeCostCoverage = "native_decision_control_flow; simulator_allocator_and_execution; scheduler_cpu_cost_not_independently_calibrated"
	}
	out.FirstTokenUS = map[string]int64{}
	out.FinishedUS = map[string]int64{}
	requestConfig := map[string]RequestConfig{}
	requestDeadlines := map[string]int64{}
	closedLoop := false
	for _, r := range c.Requests {
		requestConfig[r.ID] = r
		closedLoop = closedLoop || r.AfterRequest != ""
		if r.TTFTSLOUS > 0 {
			requestDeadlines[r.ID] = r.At + r.TTFTSLOUS
		}
	}
	if closedLoop {
		out.ArrivalUS = map[string]int64{}
	}
	sink := func(r kv.PeerRecord) { out.Events = append(out.Events, r); out.Counts[r.Name]++ }
	fabric, err := kv.NewPeerFabric(c.Resources, c.Pools, out.BlockBytes, sink)
	if err != nil {
		return nil, err
	}
	if c.Mechanisms != nil {
		if err = fabric.ConfigureMechanisms(*c.Mechanisms); err != nil {
			return nil, err
		}
	}
	if factories.PoolEviction != nil {
		for _, pool := range c.Pools {
			policy := factories.PoolEviction(pool.ID)
			if nilPolicy(policy) {
				return nil, fmt.Errorf("pool eviction factory returned nil for %q", pool.ID)
			}
			if err = fabric.SetPoolEvictionPolicy(pool.ID, policy); err != nil {
				return nil, err
			}
			out.recordInjectedPolicy("pool_eviction", pool.ID, policy)
		}
	}
	decisions := map[string]sim.DecisionPolicy{}
	if factories.Decision != nil {
		for i := range c.Instances {
			id := fmt.Sprintf("instance_%d", i)
			policy := factories.Decision(id)
			if nilPolicy(policy) {
				return nil, fmt.Errorf("decision factory returned nil for %q", id)
			}
			if storage := c.DecisionPolicy.PreemptionStorage; storage != nil {
				if _, err := storage.NewPolicy(policy); err != nil {
					return nil, err
				}
			}
			decisions[id] = policy
			out.recordInjectedPolicy("decision", id, policy)
		}
	}
	for _, r := range c.Requests {
		if r.TTFTSLOUS > 0 {
			fabric.SetRequestDeadline(r.ID, r.At+r.TTFTSLOUS)
		}
	}
	stores := make([]*kv.PeerCache, len(c.Instances))
	for i, ic := range c.Instances {
		id := fmt.Sprintf("instance_%d", i)
		var policy kv.PeerPolicy = kv.BuiltinPeerPolicy{Name: c.Policy.Name, MinFrequency: c.Policy.MinFrequency, MaxStoreUS: c.Policy.MaxStoreUS}
		if factories.Peer != nil {
			policy = factories.Peer(id)
			if nilPolicy(policy) {
				return nil, fmt.Errorf("peer factory returned nil for %q", id)
			}
			out.recordInjectedPolicy("peer", id, policy)
		}
		stores[i], err = kv.NewPeerCache(id, ic.HBMBlocks, c.BlockTokens, fabric, ic.Access, policy)
		if err != nil {
			return nil, err
		}
		if c.DecodeCapacityReservation {
			if err := stores[i].EnableDecodeCapacityReservation(); err != nil {
				return nil, err
			}
			out.DecodeCapacityCostCoverage = "client_budget_reservation_not_independently_calibrated"
		}
		if c.PrefixCopySelection == "first_published" {
			if err := stores[i].EnableFirstPublishedCopy(); err != nil {
				return nil, err
			}
			out.PrefixCopyCostCoverage = "duplicate_prefix_index_not_independently_calibrated"
		}
		if c.HotPrefix != nil {
			if err := stores[i].ConfigureHotPrefix(*c.HotPrefix); err != nil {
				return nil, err
			}
			out.HotPrefixCostCoverage = "block_granular_exact_history; declared_planner_us_only; heat_reclaim_admission_late_load_planning_and_worker_metadata_overhead_unmeasured; native_ranking_unvalidated"
		}
	}
	requests := make([]*sim.Request, 0, len(c.Requests))
	if c.EnginePhases != nil {
		if c.EnginePhases.BlockBytes != out.BlockBytes {
			return nil, fmt.Errorf("engine phase profile KV block size mismatch")
		}
		if c.EnginePhases.WorkerMetadata != nil {
			out.WorkerMetadataCostCoverage = c.EnginePhases.WorkerMetadata.Coverage
		}
		err = stores[0].ConfigureEnginePhases(*c.EnginePhases, func(start, end int64, work []sim.BatchWork) {
			if c.TraceBatchShapes && len(work) > 0 {
				row := BatchCostObservation{StartUS: start, DurationUS: end - start}
				for _, w := range work {
					row.Scheduled = append(row.Scheduled, BatchCostRequest{Request: w.Request.ID, Prefix: w.PrefixTokens, Query: w.NewTokens})
				}
				out.BatchShapes = append(out.BatchShapes, row)
			}
		})
		if err != nil {
			return nil, err
		}
		if err := stores[0].SetEnginePhaseObserver(func(row kv.EnginePhaseObservation) {
			out.EnginePhaseObservations = append(out.EnginePhaseObservations, row)
		}); err != nil {
			return nil, err
		}
		if c.DirectionalTransferOrder {
			if err := stores[0].EnableDirectionalTransferOrder(); err != nil {
				return nil, err
			}
			out.DirectionalTransferOrderCoverage = "native_handler_dependency_model_new_execution_order_not_native_rank_validated"
		}
		if hasSubmissionPolicy {
			var policy kv.TransferSubmissionPolicy
			if factories.TransferSubmission != nil {
				policy = factories.TransferSubmission("instance_0")
				if nilPolicy(policy) {
					return nil, fmt.Errorf("transfer submission factory returned nil for instance_0")
				}
				out.recordInjectedPolicy("transfer_submission", "instance_0", policy)
			} else {
				policy, err = kv.NewTransferSubmissionPolicy(c.TransferSubmissionPolicy)
				if err != nil {
					return nil, err
				}
			}
			err = stores[0].SetTransferSubmissionPolicy(policy, c.TransferSubmissionCost, func(record kv.TransferSubmissionRecord) {
				out.TransferSubmissions = append(out.TransferSubmissions, record)
			})
			if err != nil {
				return nil, err
			}
			out.TransferSubmissionCostCoverage = c.TransferSubmissionCost.Coverage()
		}
		if c.StoreSourceReuse {
			if err := stores[0].EnableStoreSourceReuse(); err != nil {
				return nil, err
			}
		}
		if c.RestoreControl {
			if err := stores[0].EnableRestoreDecisions(); err != nil {
				return nil, err
			}
			out.RestoreCostCoverage = "engine_phase_profile_restore_choices_not_native_validated"
		}
		if c.PromotionControl {
			retention := c.PromotionRetention
			if retention == "" {
				retention = "cache"
			}
			if err := stores[0].EnablePromotionDecisionsWithRetention(retention); err != nil {
				return nil, err
			}
			out.PromotionCostCoverage = "existing_whole_block_copy_phases_charged_promotion_control_not_native_validated"
		}
		if c.BatchPrefixReuse {
			if err := stores[0].EnableBatchPrefixReuse(); err != nil {
				return nil, err
			}
			out.BatchPrefixCostCoverage = "actual_batch_shape_charged_extra_lookup_validation_and_lease_work_not_independently_calibrated"
		}
	}
	if c.Native != nil {
		p, err := kvruntime.LoadProfile(c.Native.Profile)
		if err != nil {
			return nil, err
		}
		out.NativeProfile = p
		if c.Native.ComputeGrid {
			known := map[[2]int64]bool{}
			for _, shape := range p.ComputeGrid {
				known[[2]int64{shape.Prefix, shape.Query}] = true
			}
			if c.Native.PagedForward {
				known = map[[2]int64]bool{}
				for _, shape := range p.PagedForward {
					known[[2]int64{shape.Prefix, shape.Query}] = true
				}
			}
			for _, req := range c.Requests {
				length := int64(len(req.Input))
				for prefix := int64(0); prefix < length; prefix += c.BlockTokens {
					query := min(c.PrefillChunk, length-prefix)
					if !known[[2]int64{prefix, query}] {
						return nil, fmt.Errorf("request %s has an unmeasured possible prefix=%d query=%d", req.ID, prefix, query)
					}
				}
				for step := 0; step < len(req.Output)-1; step++ {
					if !known[[2]int64{length + int64(step), 1}] {
						return nil, fmt.Errorf("request %s exceeds measured decode grid", req.ID)
					}
				}
			}
		}
		if err = fabric.ConfigureNative(p, c.Native.NUMA, func(r kvruntime.Record) { out.NativeEvents = append(out.NativeEvents, r) }); err != nil {
			return nil, err
		}
		if c.Native.PagedForward {
			if err = fabric.EnablePagedForward(); err != nil {
				return nil, err
			}
		}
		if c.Native.TorchCopies {
			if err = fabric.EnableTorchCopies(); err != nil {
				return nil, err
			}
		}
		if c.Native.OutputCopies {
			policy := c.Native.CopyEnginePolicy
			if policy == "" {
				policy = "fifo"
			}
			if err = fabric.EnableOutputCopies(policy); err != nil {
				return nil, err
			}
		}
		if c.Native.InputCopies {
			var inputRequests []kvruntime.InputRequest
			for _, r := range c.Requests {
				x := kvruntime.InputRequest{ID: r.ID}
				for _, token := range r.Input {
					x.Prompt = append(x.Prompt, uint64(uint32(token)))
				}
				for _, token := range r.Output {
					x.Output = append(x.Output, uint64(uint32(token)))
				}
				inputRequests = append(inputRequests, x)
			}
			if err = fabric.EnableInputPipeline(inputRequests); err != nil {
				return nil, err
			}
		}
	}
	for _, r := range c.Requests {
		requests = append(requests, &sim.Request{ID: r.ID, ArrivalTime: r.At, InputTokens: append([]sim.TokenID(nil), r.Input...), OutputTokens: append([]sim.TokenID(nil), r.Output...), State: sim.StateQueued})
		if c.DecisionPolicy != nil {
			requests[len(requests)-1].SLOTargetUs = r.TTFTSLOUS
			requests[len(requests)-1].MaxOutputLen = r.MaxOutputTokens
		}
		if !closedLoop {
			sink(kv.PeerRecord{Time: r.At, Name: "external_arrival", Request: r.ID})
		}
	}
	sort.SliceStable(requests, func(i, j int) bool { return requests[i].ArrivalTime < requests[j].ArrivalTime })
	conversation := newConversationArrivals(c, requests, requestConfig, requestDeadlines)
	initial := requests
	if closedLoop {
		initial = nil
		for _, r := range requests {
			if requestConfig[r.ID].AfterRequest == "" {
				initial = append(initial, r)
			}
		}
	}
	scfg := sim.SimConfig{Horizon: math.MaxInt64, Seed: c.Seed, BatchCompletionEvents: true,
		KVCacheConfig:       sim.NewKVCacheConfig(c.Instances[0].HBMBlocks, c.BlockTokens, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk),
		LatencyCoeffs:       sim.NewLatencyCoeffs(nil, c.Alpha),
		ModelHardwareConfig: sim.NewModelHardwareConfig(c.Model, c.Hardware, "policy-model", "configured-gpu", 1, 1, false, "", "roofline", 0)}
	index := 0
	dcfg := cluster.DeploymentConfig{SimConfig: scfg, NumInstances: len(stores), RoutingPolicy: c.Routing, RoutingLatency: c.RoutingUS, TraceLevel: "decisions"}
	dcfg.InstanceFactory = func(id cluster.InstanceID, sc sim.SimConfig) *cluster.InstanceSimulator {
		s := stores[index]
		index++
		sc.KVCacheConfig = sim.NewKVCacheConfig(c.Instances[index-1].HBMBlocks, c.BlockTokens, 0, 0, 0, 0)
		var inst *cluster.InstanceSimulator
		if c.BatchCost != nil {
			cost := &batchCostLatency{store: s, config: *c.BatchCost}
			inst = cluster.NewInstanceSimulatorWithBackends(id, sc, s, cost)
			if c.TraceBatchShapes {
				cost.observe = func(row BatchCostObservation) {
					row.StartUS = inst.Clock()
					out.BatchShapes = append(out.BatchShapes, row)
				}
			}
		} else if c.Native != nil && c.Native.ComputeGrid {
			inst = cluster.NewInstanceSimulatorWithBackends(id, sc, s, &nativeLatency{store: s})
		} else {
			inst = cluster.NewInstanceSimulatorWithKVStore(id, sc, s)
		}
		if c.DecisionPolicy != nil {
			order := c.DecisionPolicy.QueueOrder
			if order == "" {
				order = c.decisionQueueOrder()
			}
			if order == "" {
				order = "fcfs"
			}
			policy, e := sim.NewQueueDecisionPolicy(order, c.DecisionPolicy.PrefillTokenCap)
			if policy != nil {
				policy.AdmissionDelayUS = c.DecisionPolicy.AdmissionDelayUS
				policy.RestorePrefixBlocks = c.DecisionPolicy.RestorePrefixBlocks
			}
			if e != nil {
				panic(e)
			}
			var selected sim.DecisionPolicy = policy
			if budget := c.DecisionPolicy.BudgetRestore; budget != nil {
				selected, e = budget.newPolicy(c.DecisionPolicy.PrefillTokenCap)
				if e != nil {
					panic(e)
				}
				order = "budget_restore"
			}
			if factories.Decision != nil {
				selected = decisions[string(id)]
				order = "custom_decision"
			}
			if balanced := c.DecisionPolicy.BalancedBatch; balanced != nil && factories.Decision == nil {
				selected, e = balanced.newPolicy(c.DecisionPolicy.PrefillTokenCap)
				if e != nil {
					panic(e)
				}
				order = "balanced_batch"
			}
			if selected == nil {
				panic("decision factory returned nil policy")
			}
			if storage := c.DecisionPolicy.PreemptionStorage; storage != nil {
				selected, e = storage.NewPolicy(selected)
				if e != nil {
					panic(e)
				}
			}
			if covered, ok := selected.(interface{ AdditionalCostCoverage() string }); ok {
				out.DecisionAdditionalCostCoverage = covered.AdditionalCostCoverage()
			}
			if w := c.DecisionPolicy.ProducerWait; w != nil && factories.Decision == nil {
				selected, e = sim.NewProducerWaitPolicy(selected, w.MinSharedTokens, w.MaxWaitUS)
				if e != nil {
					panic(e)
				}
				order = "producer_wait"
			}
			// The probe changes decisions only; retain the selected base's
			// existing lifecycle subscription without adding a new one.
			listener, subscribed := selected.(sim.DecisionEventListener)
			if probe := c.DecisionPolicy.PrefillWaitProbe; probe != nil {
				selected, e = probe.NewPolicy(selected)
				if e != nil {
					panic(e)
				}
			}
			adapter := &decisionAdapter{policy: selected, order: order, instance: string(id)}
			if probe := c.DecisionPolicy.PrefillDeferralProbe; probe != nil {
				selected, e = probe.NewPolicy(selected)
				if e != nil {
					panic(e)
				}
				adapter.policy = selected
				out.DecisionAdditionalCostCoverage = "prefill_deferral_probe_not_independently_calibrated"
			}
			if c.RequestScheduler != "" {
				adapter.sink = sink
			}
			var installed sim.DecisionPolicy = adapter
			if subscribed {
				installed = &eventDecisionAdapter{decisionAdapter: adapter, listener: listener}
			}
			if e := inst.SetDecisionPolicy(installed, func(r sim.DecisionRecord) {
				r.Instance = string(id)
				out.PolicyDecisions = append(out.PolicyDecisions, r)
			}); e != nil {
				panic(e)
			}
			out.DecisionCostCoverage = "host_measured_sim_time_uncalibrated"
			if c.DecisionPolicy.WaitPostStepCost != nil {
				model, err := newWaitPostStepCost(c)
				if err != nil {
					panic(err)
				}
				if err := inst.SetPostStepCostModel(model, len(c.Requests), func(r sim.PostStepRecord) {
					r.Instance = string(id)
					out.PostStepServices = append(out.PostStepServices, r)
				}); err != nil {
					panic(err)
				}
				out.PostStepCostCoverage = "wait_after_output_bookkeeping_and_idle_only_pre_step_driver_and_residuals_uncovered"
				if model.DriverLoopCostsEnabled() {
					out.PostStepCostCoverage = "wait_driver_loop_and_post_output_idle_services_residuals_and_wakeup_uncovered"
				}
			}
			if c.DecisionPolicy.HostServiceCost != nil {
				model, err := newCapacityHostServiceCost(c)
				if err != nil {
					panic(err)
				}
				profile := c.DecisionPolicy.HostServiceCost
				if err := inst.SetHostServiceCosts(model, profile.CompletionRatesUS != nil, profile.RegistrationUS != nil, func(r sim.HostServiceRecord) { r.Instance = string(id); out.HostServices = append(out.HostServices, r) }); err != nil {
					panic(err)
				}
				out.HostServiceCostCoverage = "explicit_capacity_completion_registration_service_worker_and_native_residuals_uncovered"
			}
			if c.DecisionPolicy.WaitHostServiceCost != nil {
				model, err := newWaitHostServiceCost(c)
				if err != nil {
					panic(err)
				}
				profile := c.DecisionPolicy.WaitHostServiceCost
				if err := inst.SetHostServiceCosts(model, profile.CompletionRatesUS != nil, profile.RegistrationUS != nil, func(r sim.HostServiceRecord) { r.Instance = string(id); out.HostServices = append(out.HostServices, r) }); err != nil {
					panic(err)
				}
				out.HostServiceCostCoverage = "wait_host"
				if profile.CompletionRatesUS != nil {
					out.HostServiceCostCoverage += "_additional_completion"
				}
				if profile.RegistrationUS != nil {
					out.HostServiceCostCoverage += "_full_registration_replaces_enqueue"
				}
				out.HostServiceCostCoverage += "_other_services_declared_separately"
			}
			if c.DecisionPolicy.ExecutionCost != nil {
				model, err := newCapacityExecutionCost(c)
				if err != nil {
					panic(err)
				}
				if err := inst.SetExecutionCostModel(model); err != nil {
					panic(err)
				}
			}
			if c.DecisionPolicy.WaitExecutionCost != nil {
				model, err := newWaitExecutionCost(c)
				if err != nil {
					panic(err)
				}
				if err := inst.SetExecutionCostModel(model); err != nil {
					panic(err)
				}
				out.WaitExecutionCostCoverage = "wait_execution_wrapper_only_other_services_declared_separately"
			}
			if c.DecisionPolicy.ExecutionWork {
				if err := inst.EnableExecutionWorkObservation(); err != nil {
					panic(err)
				}
			}
			if c.DecisionPolicy.PrefillPreemption {
				if err := inst.EnableDecisionPrefillPreemption(); err != nil {
					panic(err)
				}
				out.PrefillPreemptionCostCoverage = "runtime_action_not_independently_calibrated"
			}
			if c.DecisionPolicy.PrefillWaits {
				if err := inst.EnableDecisionPrefillWaits(); err != nil {
					panic(err)
				}
				out.PrefillWaitCostCoverage = "resident_prefill_wait_and_timer_work_not_independently_calibrated"
			}
			if c.DecisionPolicy.PrefillDeferrals {
				if err := inst.EnableDecisionPrefillDeferrals(); err != nil {
					panic(err)
				}
				out.PrefillDeferralCostCoverage = "resident_prefill_one_round_deferral_not_independently_calibrated"
			}
			if c.DecisionPolicy.CapacityPreemption {
				if err := inst.EnableDecisionCapacityPreemption(); err != nil {
					panic(err)
				}
				out.PrefillPreemptionCostCoverage = "capacity_failure_and_action_not_independently_calibrated"
			}
			if c.DecisionPolicy.RequestSpill {
				if err := inst.EnableDecisionRequestSpill(); err != nil {
					panic(err)
				}
				out.RequestSpillCostCoverage = "request_spill_host_actions_not_independently_calibrated"
			}
			if c.DecisionPolicy.CapacityReservationView {
				if err := inst.EnableDecisionCapacityReservationView(); err != nil {
					panic(err)
				}
				out.CapacityReservationViewCostCoverage = "current_reservation_view_not_independently_calibrated"
			}
			if c.DecisionPolicy.PrefixSharing {
				if err := inst.EnableDecisionPrefixSharing(); err != nil {
					panic(err)
				}
			}
			if c.DecisionPolicy.ProducerPrefixes {
				if err := inst.EnableDecisionPrefixProducers(); err != nil {
					panic(err)
				}
			}
			if c.DecisionPolicy.ControlSteps {
				if err := inst.EnableDecisionControlSteps(); err != nil {
					panic(err)
				}
			}
			if c.DecisionEstimates != nil {
				estimator, err := newRestoreEstimator(c)
				if err != nil {
					panic(err)
				}
				if err = inst.SetDecisionEstimator(estimator); err != nil {
					panic(err)
				}
				out.DecisionEstimateCoverage = RestoreEstimateCoverage
				if c.DecisionEstimates.Spill {
					out.SpillEstimateCoverage = SpillEstimateCoverage
				}
			}
			if subscribed || c.DecisionPolicy.TraceEvents {
				if e := inst.SetDecisionEventObserver(func(r sim.DecisionEventRecord) {
					r.Instance = string(id)
					out.PolicyEvents = append(out.PolicyEvents, r)
				}); e != nil {
					panic(e)
				}
				out.DecisionEventCostCoverage = sim.DecisionEventCostCoverage
			}
			if c.DecisionPolicy.ExtraCost != nil {
				model := *c.DecisionPolicy.ExtraCost
				model.SetProfileWarningObserver(c.profile("linear_decision", model.Provenance).above)
				if e := inst.SetDecisionCostModel(model); e != nil {
					panic(e)
				}
				out.DecisionCostCoverage = sim.ExplicitDecisionCostCoverage
			}
			if c.DecisionPolicy.WaitDecisionCost != nil {
				model, err := newWaitDecisionCost(c)
				if err != nil {
					panic(err)
				}
				if err := inst.SetDecisionCostModel(model); err != nil {
					panic(err)
				}
				out.DecisionCostCoverage = "wait_decision_wrapper_only_other_services_declared_separately"
			}
			if c.DecisionPolicy.RestoreServiceCost != nil {
				model, err := newRestoreServiceCost(c)
				if err != nil {
					panic(err)
				}
				if err := inst.SetDecisionCostModel(model); err != nil {
					panic(err)
				}
				out.DecisionCostCoverage = c.DecisionPolicy.RestoreServiceCost.Coverage
			}
			if c.DecisionPolicy.QueueServiceCost != nil {
				model, err := newQueueServiceCost(c)
				if err != nil {
					panic(err)
				}
				if err := inst.SetDecisionCostModel(model); err != nil {
					panic(err)
				}
				if err := inst.SetExecutionOutcomeCostModel(model); err != nil {
					panic(err)
				}
				if err := inst.SetHostServiceCosts(model, true, true, func(r sim.HostServiceRecord) { r.Instance = string(id); out.HostServices = append(out.HostServices, r) }); err != nil {
					panic(err)
				}
				out.QueueServiceCostCoverage = model.profile.Coverage
				if model.profile.Spill != nil {
					out.RequestSpillCostCoverage = model.profile.Spill.Coverage
				}
				out.DecisionCostCoverage = "queue_decision_reservation_estimate_separate_from_dispatch"
				out.HostServiceCostCoverage = "queue_control_completion_and_full_registration_replacing_enqueue"
				out.CapacityReservationViewCostCoverage = "queue_reservation_component_in_decision_service"
			}
			if c.DecisionPolicy.DecodeFillServiceCost != nil {
				model, err := newDecodeFillServiceCost(c)
				if err != nil {
					panic(err)
				}
				if err = inst.SetDecisionCostModel(model); err != nil {
					panic(err)
				}
				if err = inst.SetExecutionOutcomeCostModel(model); err != nil {
					panic(err)
				}
				out.DecodeFillServiceCostCoverage = model.profile.Coverage
				out.DecisionCostCoverage = "explicit_controller_prepare_separate_from_execution"
				out.DecisionAdditionalCostCoverage = model.profile.Coverage
				out.PrefillDeferralCostCoverage = "controller_profile_covers_scoped_deferral_work; see declared coverage"
				out.PromotionCostCoverage = "controller_profile_excludes_engine_owned_promotion_work; base engine fees retained"
			}
		} else if c.RequestScheduler != "" && c.RequestScheduler != "vllm_native" {
			inst.SetInstanceScheduler(&requestScheduler{name: c.RequestScheduler, instance: string(id), deadlines: requestDeadlines, sink: sink})
		}
		if c.RequestScheduler == "vllm_native" {
			if err := inst.SetVLLMNativeScheduler(); err != nil {
				panic(err) // validated before instance construction
			}
		}
		var started time.Time
		inst.SetEventObserver(func(e sim.Event, before bool) {
			name := strings.TrimPrefix(fmt.Sprintf("%T", e), "*")
			if before {
				started = time.Now()
				r := kv.PeerRecord{Time: e.Timestamp(), Name: name, Instance: string(id)}
				switch v := e.(type) {
				case *sim.ArrivalEvent:
					r.Request = v.Request.ID
				case *sim.QueuedEvent:
					r.Request = v.Request.ID
				case *sim.RequestLeftEvent:
					r.Request = v.Request.ID
					out.FinishedUS[v.Request.ID] = e.Timestamp()
				case *sim.ScheduledEvent:
					r.Request = v.Request.ID
				case *sim.BatchCompleteEvent:
					r.Name = "compute"
					r.Time = v.Start
					r.Duration = v.Duration
					r.Requests = v.Requests
				}
				sink(r)
			} else {
				if done, ok := e.(*sim.BatchCompleteEvent); ok {
					for _, rid := range done.Requests {
						if _, seen := out.FirstTokenUS[rid]; seen || len(requestConfig[rid].Output) == 0 {
							continue
						}
						if ttft, emitted := inst.Metrics().RequestTTFTs[rid]; emitted {
							out.FirstTokenUS[rid] = requestConfig[rid].At + int64(ttft)
							sink(kv.PeerRecord{Time: out.FirstTokenUS[rid], Name: "first_token_observed", Instance: string(id), Request: rid})
						}
					}
				}
				p := out.CPU[name]
				p.Count++
				p.Nanoseconds += time.Since(started).Nanoseconds()
				out.CPU[name] = p
				m := s.PeerSnapshot()
				sink(kv.PeerRecord{Time: e.Timestamp(), Name: "hbm_capacity", Instance: string(id), Counters: m})
				pools := fabric.Snapshot()
				ids := make([]string, 0, len(pools))
				for p := range pools {
					ids = append(ids, p)
				}
				sort.Strings(ids)
				for _, p := range ids {
					sink(kv.PeerRecord{Time: e.Timestamp(), Name: "l2_capacity", Destination: p, Counters: pools[p]})
				}
				if c.Mechanisms != nil && (c.Mechanisms.DirectoryUS > 0 || c.Mechanisms.CompletionUS > 0 || c.Mechanisms.PolicyUS > 0) {
					sink(kv.PeerRecord{Time: e.Timestamp(), Name: "control_capacity", Destination: "control", Counters: fabric.ControlSnapshot()})
				}
			}
		})
		return inst
	}
	var onDone func(*sim.Request, int64) []*sim.Request
	if closedLoop {
		onDone = conversation.complete
	}
	cs := cluster.NewClusterSimulator(dcfg, cluster.NewSliceRequestSource(initial), onDone)
	if closedLoop {
		cs.SetArrivalHook(func(r *sim.Request) {
			out.ArrivalUS[r.ID] = r.ArrivalTime
			sink(kv.PeerRecord{Time: r.ArrivalTime, Name: "external_arrival", Request: r.ID})
		})
	}
	for _, s := range stores {
		s.StartOnlinePromotion()
	}
	for _, p := range c.Promotions {
		stores[p.Instance].SchedulePromotion(p.At, p.Tokens, p.Budget)
	}
	if err := cs.Run(); err != nil {
		out.CacheMetrics = collectCacheMetrics(c, stores)
		return out, err
	}
	out.CacheMetrics = collectCacheMetrics(c, stores)
	if conversation.err != nil {
		return out, conversation.err
	}
	for _, request := range requests {
		if request.PrefillEnd() > request.InputLen() {
			out.RequestRecomputeCostCoverage = "compute_and_transfer_charged_decode_recompute_host_path_not_native_validated"
			break
		}
	}
	if err := fabric.NativeCheck(); err != nil {
		return out, err
	}
	out.NativeValidated = c.Native != nil
	out.NativeAllocations = fabric.NativeAllocationSnapshot()
	m := cs.AggregatedMetrics()
	out.Metrics = m.BuildOutput("cluster")
	out.Requests = m.CompletedRequestMetrics()
	for i := range out.Requests {
		q := &out.Requests[i]
		q.ITL = m.RequestITLs[q.ID] / 1000
		q.SchedulingDelay = float64(m.RequestSchedulingDelays[q.ID]) / 1000
	}
	out.Pools = fabric.Snapshot()
	out.RoutingTrace = cs.Trace()
	for i, s := range stores {
		out.HBM[fmt.Sprintf("instance_%d", i)] = s.PeerSnapshot()
	}
	// Sort only by simulated time; stable ties retain execution order.
	sort.SliceStable(out.Events, func(i, j int) bool { return out.Events[i].Time < out.Events[j].Time })
	if m.CompletedRequests != len(requests) || fabric.Pending() != 0 || m.KVAllocationFailures != 0 {
		return out, fmt.Errorf("incomplete run: completed=%d/%d transfers_pending=%d kv_failures=%d", m.CompletedRequests, len(requests), fabric.Pending(), m.KVAllocationFailures)
	}
	for id, h := range out.HBM {
		if h["active_or_pinned"] != 0 || h["held_restores"] != 0 || h["worker_resident_requests"] != 0 {
			return out, fmt.Errorf("HBM references remain on %s: %v", id, h)
		}
	}
	return out, nil
}
func Decode(data []byte) (Config, error) {
	var c Config
	d := json.NewDecoder(strings.NewReader(string(data)))
	d.DisallowUnknownFields()
	err := d.Decode(&c)
	return c, err
}
