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
	DecisionPolicy   *DecisionPolicyConfig  `json:"decision_policy,omitempty"`
	EnginePhases     *EnginePhaseCostConfig `json:"engine_phases,omitempty"`
	StoreSourceReuse bool                   `json:"store_source_reuse,omitempty"`
	TraceBatchShapes bool                   `json:"trace_batch_shapes,omitempty"`
	BatchCost        *BatchCostConfig       `json:"batch_cost,omitempty"`
	Native           *NativeConfig          `json:"native_runtime,omitempty"`
	Seed             int64                  `json:"seed"`
	RequestScheduler string                 `json:"request_scheduler,omitempty"`
	BlockTokens      int64                  `json:"block_tokens"`
	MaxBatchTokens   int64                  `json:"max_batch_tokens"`
	MaxSequences     int64                  `json:"max_sequences"`
	PrefillChunk     int64                  `json:"prefill_chunk"`
	Model            sim.ModelConfig        `json:"model"`
	Hardware         sim.HardwareCalib      `json:"hardware"`
	Alpha            []float64              `json:"alpha_us"`
	Routing          string                 `json:"routing"`
	RoutingUS        int64                  `json:"routing_us"`
	Resources        []kv.PeerResource      `json:"resources"`
	Pools            []kv.PeerPoolConfig    `json:"pools"`
	Instances        []InstanceConfig       `json:"instances"`
	Policy           PolicyConfig           `json:"policy"`
	Requests         []RequestConfig        `json:"requests"`
	Promotions       []PromotionConfig      `json:"promotions"`
	Mechanisms       *kv.PeerMechanisms     `json:"mechanisms,omitempty"`
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
	PolicyDecisions      []sim.DecisionRecord   `json:"-"`
	DecisionCostCoverage string                 `json:"decision_cost_coverage,omitempty"`
	BatchShapes          []BatchCostObservation `json:"batch_shapes,omitempty"`
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
	if c.DecisionPolicy != nil {
		if c.DecisionPolicy.PrefillTokenCap > c.MaxBatchTokens {
			return fmt.Errorf("decision prefill cap exceeds batch token limit")
		}
		order := c.DecisionPolicy.QueueOrder
		if order == "" {
			order = c.RequestScheduler
		}
		if _, err := sim.NewQueueDecisionPolicy(order, c.DecisionPolicy.PrefillTokenCap); err != nil {
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
	case "", "fcfs", "sjf", "edf":
	default:
		return fmt.Errorf("request_scheduler must be fcfs, sjf or edf")
	}
	if c.Mechanisms != nil {
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
	case "lru_drop", "lfu_store", "ready_first", "cost_aware":
	default:
		return fmt.Errorf("unknown peer policy %q", c.Policy.Name)
	}
	if c.Policy.MinFrequency < 0 || c.Policy.MaxStoreUS < 0 {
		return fmt.Errorf("policy thresholds must be nonnegative")
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
	return run(c, nil)
}

// RunWithDecisionPolicy lets experiments inject a fresh stateful policy per
// instance without changing resource management or event/metric code. The
// decision_policy config must be enabled; its built-in ordering/cap settings
// are replaced by the factory's implementation.
func RunWithDecisionPolicy(c Config, factory func(instance string) sim.DecisionPolicy) (*Result, error) {
	if c.DecisionPolicy == nil || factory == nil {
		return nil, fmt.Errorf("custom decision factory requires decision_policy and a nonnil factory")
	}
	return run(c, factory)
}

func run(c Config, decisionFactory func(string) sim.DecisionPolicy) (*Result, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	bytes, err := latency.KVBytesPerToken(c.Model, 1)
	if err != nil {
		return nil, err
	}
	out := &Result{Config: c, BlockBytes: int64(math.Ceil(bytes * float64(c.BlockTokens))), Counts: map[string]int64{}, CPU: map[string]CPUProfile{}, HBM: map[string]map[string]int64{}}
	out.FirstTokenUS = map[string]int64{}
	out.FinishedUS = map[string]int64{}
	requestConfig := map[string]RequestConfig{}
	for _, r := range c.Requests {
		requestConfig[r.ID] = r
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
	for _, r := range c.Requests {
		if r.TTFTSLOUS > 0 {
			fabric.SetRequestDeadline(r.ID, r.At+r.TTFTSLOUS)
		}
	}
	stores := make([]*kv.PeerCache, len(c.Instances))
	for i, ic := range c.Instances {
		stores[i], err = kv.NewPeerCache(fmt.Sprintf("instance_%d", i), ic.HBMBlocks, c.BlockTokens, fabric, ic.Access, kv.BuiltinPeerPolicy{Name: c.Policy.Name, MinFrequency: c.Policy.MinFrequency, MaxStoreUS: c.Policy.MaxStoreUS})
		if err != nil {
			return nil, err
		}
	}
	requests := make([]*sim.Request, 0, len(c.Requests))
	if c.EnginePhases != nil {
		if c.EnginePhases.BlockBytes != out.BlockBytes {
			return nil, fmt.Errorf("engine phase profile KV block size mismatch")
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
		if c.StoreSourceReuse {
			if err := stores[0].EnableStoreSourceReuse(); err != nil {
				return nil, err
			}
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
		sink(kv.PeerRecord{Time: r.At, Name: "external_arrival", Request: r.ID})
	}
	sort.SliceStable(requests, func(i, j int) bool { return requests[i].ArrivalTime < requests[j].ArrivalTime })
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
				order = c.RequestScheduler
			}
			if order == "" {
				order = "fcfs"
			}
			policy, e := sim.NewQueueDecisionPolicy(order, c.DecisionPolicy.PrefillTokenCap)
			if e != nil {
				panic(e)
			}
			var selected sim.DecisionPolicy = policy
			if decisionFactory != nil {
				selected = decisionFactory(string(id))
				order = "custom_decision"
			}
			if selected == nil {
				panic("decision factory returned nil policy")
			}
			adapter := &decisionAdapter{policy: selected, order: order, instance: string(id)}
			if c.RequestScheduler != "" {
				adapter.sink = sink
			}
			if e := inst.SetDecisionPolicy(adapter, func(r sim.DecisionRecord) {
				r.Instance = string(id)
				out.PolicyDecisions = append(out.PolicyDecisions, r)
			}); e != nil {
				panic(e)
			}
			out.DecisionCostCoverage = "host_measured_sim_time_uncalibrated"
		} else if c.RequestScheduler != "" {
			deadlines := make(map[string]int64)
			for _, r := range c.Requests {
				if r.TTFTSLOUS > 0 {
					deadlines[r.ID] = r.At + r.TTFTSLOUS
				}
			}
			inst.SetInstanceScheduler(&requestScheduler{name: c.RequestScheduler, instance: string(id), deadlines: deadlines, sink: sink})
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
	cs := cluster.NewClusterSimulator(dcfg, cluster.NewSliceRequestSource(requests), nil)
	for _, s := range stores {
		s.StartOnlinePromotion()
	}
	for _, p := range c.Promotions {
		stores[p.Instance].SchedulePromotion(p.At, p.Tokens, p.Budget)
	}
	if err := cs.Run(); err != nil {
		return out, err
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
		if h["active_or_pinned"] != 0 || h["held_restores"] != 0 {
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
