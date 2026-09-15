package sim

import (
	"fmt"
	"sort"
	"time"
)

// DecisionRequest is a detached view of an arrived request. ClientOutputLimit
// is an explicit client budget; actual future output tokens/length are absent.
type DecisionRequest struct {
	PrefillDeferrable    bool         `json:"prefill_deferrable,omitempty"`
	RecomputeUntilTokens int64        `json:"recompute_until_tokens,omitempty"`
	PrefillWaitable      bool         `json:"prefill_waitable,omitempty"`
	PrefillPreemptible   bool         `json:"prefill_preemptible,omitempty"`
	ID                   string       `json:"id"`
	State                RequestState `json:"state"`
	ArrivalUS            int64        `json:"arrival_us"`
	InputTokens          int64        `json:"input_tokens"`
	ComputedTokens       int64        `json:"computed_tokens"`
	EmittedTokens        int64        `json:"emitted_tokens"`
	ClientOutputLimit    int          `json:"client_output_limit"`
	TTFTTargetUS         int64        `json:"ttft_target_us"`
	DeadlineUS           int64        `json:"deadline_us"`
	Priority             float64      `json:"priority"`
	WaitUntilUS          int64        `json:"wait_until_us"`
}

type DecisionKVRequest struct {
	SpillTargets            []DecisionSpillTarget        `json:"spill_targets,omitempty"`
	CapacityReservation     *DecisionCapacityReservation `json:"capacity_reservation,omitempty"`
	WorkerResident          bool                         `json:"worker_resident,omitempty"`
	PrefixBlocks            []DecisionPrefixBlock        `json:"prefix_blocks,omitempty"`
	LocalPrefixBlocks       int64                        `json:"local_prefix_blocks,omitempty"`
	RecoverablePrefixBlocks int64                        `json:"recoverable_prefix_blocks,omitempty"`
	RestoreLimitSet         bool                         `json:"restore_limit_set,omitempty"`
	RestoreLimitBlocks      int64                        `json:"restore_limit_blocks,omitempty"`
	ID                      string                       `json:"id"`
	LocalPrefixTokens       int64                        `json:"local_prefix_tokens"`
	RecoverablePrefixTokens int64                        `json:"recoverable_prefix_tokens"`
	TransferPending         bool                         `json:"transfer_pending"`
	Deferred                bool                         `json:"deferred"`
}
type DecisionPool struct {
	ID              string `json:"id"`
	CapacityBlocks  int64  `json:"capacity_blocks"`
	FreeBlocks      int64  `json:"free_blocks"`
	ReadyBlocks     int64  `json:"ready_blocks"`
	ReservedBlocks  int64  `json:"reserved_blocks"`
	ProtectedBlocks int64  `json:"protected_blocks"`
}
type DecisionTransfer struct {
	ID          int64  `json:"id"`
	Request     string `json:"request"`
	Source      string `json:"source"`
	Destination string `json:"destination"`
	Bytes       int64  `json:"bytes"`
}
type DecisionKVState struct {
	WorkerStateKnown   bool                `json:"worker_state_known,omitempty"`
	PrefixSharingKnown bool                `json:"prefix_sharing_known,omitempty"`
	PrefixStateKnown   bool                `json:"prefix_state_known"`
	TransfersKnown     bool                `json:"transfers_known"`
	Requests           []DecisionKVRequest `json:"requests"`
	Pools              []DecisionPool      `json:"pools"`
	PendingTransfers   []DecisionTransfer  `json:"pending_transfers"`
}

// DecisionKVQuery is for the trusted storage runtime, never sent to a policy.
type DecisionKVQuery struct {
	SpillComputedTokens int64
	ID                  string
	Input               []TokenID
	CapacityReservation bool
	InputTokens         int64
	ClientOutputLimit   int64
}
type DecisionStateProvider interface {
	DecisionState([]DecisionKVQuery) DecisionKVState
}

type DecisionCapabilities struct {
	PrefillDeferrals      bool   `json:"prefill_deferrals,omitempty"`
	RequestSpill          bool   `json:"request_spill,omitempty"`
	CapacityReservations  bool   `json:"capacity_reservations,omitempty"`
	PrefillWaits          bool   `json:"prefill_waits,omitempty"`
	CapacityPreemption    bool   `json:"capacity_preemption,omitempty"`
	PrefillPreemption     bool   `json:"prefill_preemption,omitempty"`
	BatchTokenCap         bool   `json:"batch_token_cap,omitempty"`
	RestoreDefersCompute  bool   `json:"restore_defers_compute,omitempty"`
	PromotionRetention    string `json:"promotion_retention,omitempty"`
	Promotions            bool   `json:"promotions,omitempty"`
	BatchPrefixReuseKnown bool   `json:"batch_prefix_reuse_known,omitempty"`
	BatchPrefixReuse      bool   `json:"batch_prefix_reuse,omitempty"`
	PrefixProducers       bool   `json:"prefix_producers,omitempty"`
	AdmissionSelection    bool   `json:"admission_selection"`
	QueueOrder            bool   `json:"queue_order"`
	TokenCaps             bool   `json:"token_caps"`
	ReadyRestoresFirst    bool   `json:"ready_restores_first"`
	RestoreChoice         bool   `json:"restore_choice"`
	Placement             bool   `json:"placement"`
	TransferPriority      bool   `json:"transfer_priority"`
	CostEstimates         bool   `json:"cost_estimates"`
	LifecycleEvents       bool   `json:"lifecycle_events"`
	TransferEvents        bool   `json:"transfer_events"`
	WaitUntil             bool   `json:"wait_until"`
	Revisit               bool   `json:"revisit"`
}
type DecisionView struct {
	PrefixProducers   []DecisionPrefixProducer `json:"prefix_producers,omitempty"`
	Estimates         *DecisionEstimates       `json:"estimates,omitempty"`
	RevisitAtUS       int64                    `json:"revisit_at_us"`
	Version           uint64                   `json:"version"`
	NowUS             int64                    `json:"now_us"`
	Waiting           []DecisionRequest        `json:"waiting"`
	Running           []DecisionRequest        `json:"running"`
	MaxBatchTokens    int64                    `json:"max_batch_tokens"`
	MaxSequences      int64                    `json:"max_sequences"`
	PrefillChunk      int64                    `json:"prefill_chunk"`
	BlockTokens       int64                    `json:"block_tokens"`
	HBMCapacityBlocks int64                    `json:"hbm_capacity_blocks"`
	HBMFreeBlocks     int64                    `json:"hbm_free_blocks"`
	KV                DecisionKVState          `json:"kv"`
	Capabilities      DecisionCapabilities     `json:"capabilities"`
}

type DecisionTokenCap struct {
	Request string `json:"request"`
	Tokens  int64  `json:"tokens"`
}

// QueueOrder is nil for the runtime's current order, or a complete permutation
// of Waiting. TokenCaps are positive upper bounds, not promised token grants.
// Waits update admission deadlines, or resident prefill waits when enabled;
// RevisitAtUS schedules one
// instance revisit. Omitted updates preserve state; explicit zero cancels it.
type DecisionPlan struct {
	// PrefillDeferrals hold resident prefill computation for this round only.
	// KV ownership remains with the runtime; omission clears the prior round.
	PrefillDeferrals  []string                    `json:"prefill_deferrals,omitempty"`
	PreemptionStorage []DecisionPreemptionStorage `json:"preemption_storage,omitempty"`
	CapacityVictims   *DecisionCapacityVictims    `json:"capacity_victims,omitempty"`
	// Preemptions names running prefills to yield at this completed chunk boundary.
	// They rejoin the queue, in listed order, but cannot compute again this step.
	// QueueOrder/Admission/Waits/Restores continue to address the observed Waiting.
	Preemptions []string `json:"preemptions,omitempty"`
	// BatchTokenCap limits actual total grants for this step. Zero inherits
	// the configured limit; individual TokenCaps may sum to more than this cap.
	BatchTokenCap int64               `json:"batch_token_cap,omitempty"`
	Promotions    []DecisionPromotion `json:"promotions,omitempty"`
	Admission     *DecisionAdmission  `json:"admission,omitempty"`
	Restores      []DecisionRestore   `json:"restores,omitempty"`
	Version       uint64              `json:"version"`
	QueueOrder    []string            `json:"queue_order,omitempty"`
	TokenCaps     []DecisionTokenCap  `json:"token_caps,omitempty"`
	Waits         []DecisionWait      `json:"waits,omitempty"`
	RevisitAtUS   *int64              `json:"revisit_at_us,omitempty"`
}

// DecisionAdmission restricts waiting-request admission for this step only.
// A nil plan.Admission preserves the runtime's choice; a present empty object
// admits none. Existing running work and in-flight transfers are not cancelled.
type DecisionAdmission struct {
	Requests []string `json:"requests"`
}
type DecisionWait struct {
	Request string `json:"request"`
	UntilUS int64  `json:"until_us"`
}

// MaxPrefixBlocks is the absolute prefix endpoint for future L2 loading.
// Zero skips further loading but retains resident hits. Omission keeps the
// existing choice. It is an upper bound, not a promise that blocks will load.
type DecisionRestore struct {
	Request         string `json:"request"`
	MaxPrefixBlocks int64  `json:"max_prefix_blocks"`
}
type DecisionWaitOutcome struct {
	Request string `json:"request,omitempty"`
	UntilUS int64  `json:"until_us"`
	Status  string `json:"status"`
}
type DecisionGrant struct {
	Request string `json:"request"`
	Tokens  int64  `json:"tokens"`
}
type DecisionUnselected struct {
	Request string `json:"request"`
	Reason  string `json:"reason"`
}
type DecisionFeedback struct {
	PreemptionStorage []RequestSpillOutcome       `json:"preemption_storage,omitempty"`
	Capacity          []DecisionCapacityOutcome   `json:"capacity,omitempty"`
	Preemptions       []DecisionPreemptionOutcome `json:"preemptions,omitempty"`
	Promotions        []DecisionPromotionOutcome  `json:"promotions,omitempty"`
	RestoreUpdates    []DecisionRestore           `json:"restore_updates,omitempty"`
	Version           uint64                      `json:"version"`
	AtUS              int64                       `json:"at_us"`
	Status            string                      `json:"status"`
	Reason            string                      `json:"reason,omitempty"`
	Grants            []DecisionGrant             `json:"grants,omitempty"`
	Unselected        []DecisionUnselected        `json:"unselected,omitempty"`
	WaitUpdates       []DecisionWaitOutcome       `json:"wait_updates,omitempty"`
	Revisit           *DecisionWaitOutcome        `json:"revisit,omitempty"`
}
type DecisionRecord struct {
	ExecutionService *DecisionService    `json:"execution_service,omitempty"`
	ExecutionWork    *BatchExecutionWork `json:"execution_work,omitempty"`
	ControlStep      bool                `json:"control_step,omitempty"`
	Instance         string              `json:"instance,omitempty"`
	View             DecisionView        `json:"view"`
	Plan             DecisionPlan        `json:"plan"`
	Feedback         DecisionFeedback    `json:"feedback"`
	SnapshotWallNS   int64               `json:"snapshot_wall_ns"`
	DecisionWallNS   int64               `json:"decision_wall_ns"`
	FeedbackWallNS   int64               `json:"feedback_wall_ns"`
	RuntimeWallNS    int64               `json:"runtime_wall_ns"`
	CostCoverage     string              `json:"cost_coverage"`
	Service          *DecisionService    `json:"service,omitempty"`
}
type DecisionPolicy interface {
	Decide(DecisionView) DecisionPlan
	Observe(DecisionFeedback)
}
type decisionRuntime struct {
	prefillDeferrals        bool
	deferredPrefills        map[string]bool
	capacityReservationView bool
	prefillWaits            bool
	batchWaits              map[*Request]int64
	executionCost           ExecutionCostModel
	executionOutcomeCost    ExecutionOutcomeCostModel
	executionFeedback       *DecisionFeedback
	executionPending        *ExecutionCompleteEvent
	executionWork           bool
	capacityPreemption      bool
	capacityVictims         *DecisionCapacityVictims
	prefillPreemption       bool
	prefillPreemptions      []*Request
	requestSpill            bool
	preemptionStorage       []DecisionPreemptionStorage
	batchTokenCap           int64
	prefixSharing           bool
	prefixProducers         bool
	admission               map[string]bool
	controlSteps            bool
	controlPending          bool
	estimator               DecisionEstimator
	policy                  DecisionPolicy
	observe                 func(DecisionRecord)
	record                  DecisionRecord
	cost                    DecisionCostModel
	pending                 *DecisionCompleteEvent
	events                  *decisionEvents
	applying                bool
	waits                   map[string]decisionWaitState
	revisitAt               int64
	timer                   *DecisionTimerEvent
}

func (s *Simulator) SetDecisionPolicy(policy DecisionPolicy, observe func(DecisionRecord)) error {
	if policy == nil || s.stepCount != 0 {
		return fmt.Errorf("decision policy must be installed before execution")
	}
	if _, ok := s.batchFormation.(*VLLMBatchFormation); !ok {
		return fmt.Errorf("decision token caps require VLLMBatchFormation")
	}
	s.decision = &decisionRuntime{policy: policy, observe: observe}
	if listener, ok := policy.(DecisionEventListener); ok {
		if err := s.enableDecisionEvents(); err != nil {
			s.decision = nil
			return err
		}
		s.decision.events.listener = listener
	}
	return nil
}

func (s *Simulator) decisionView(now int64) DecisionView {
	v := DecisionView{Version: uint64(s.stepCount), NowUS: now, MaxBatchTokens: s.maxNumBatchedTokens, MaxSequences: s.maxNumSeqs, PrefillChunk: s.longPrefillTokenThreshold,
		BlockTokens: s.KVCache.BlockSize(), HBMCapacityBlocks: s.KVCache.TotalCapacity(), HBMFreeBlocks: s.KVCache.TotalCapacity() - s.KVCache.UsedBlocks(),
		Capabilities: DecisionCapabilities{QueueOrder: true, TokenCaps: true, AdmissionSelection: true, BatchTokenCap: true}}
	_, v.Capabilities.ReadyRestoresFirst = s.KVCache.(DeferredAdmissionStore)
	if backend, ok := s.KVCache.(PromotionDecisionStore); ok {
		v.Capabilities.Promotions = backend.PromotionDecisionsEnabled()
		if v.Capabilities.Promotions {
			v.Capabilities.PromotionRetention = backend.PromotionRetention()
		}
	}
	if backend, ok := s.KVCache.(BatchPrefixStore); ok {
		v.Capabilities.BatchPrefixReuseKnown = true
		v.Capabilities.BatchPrefixReuse = backend.BatchPrefixReuseEnabled()
	}
	if backend, ok := s.KVCache.(RestoreDecisionStore); ok {
		v.Capabilities.RestoreChoice = backend.RestoreDecisionsEnabled()
	}
	if backend, ok := s.KVCache.(interface{ RestoreDecisionsDeferCompute() bool }); ok {
		v.Capabilities.RestoreDefersCompute = backend.RestoreDecisionsDeferCompute()
	}
	v.Capabilities.WaitUntil, v.Capabilities.Revisit = s.batchCompletionEvents, s.batchCompletionEvents
	if s.decision != nil {
		v.RevisitAtUS = s.decision.revisitAt
		v.Capabilities.CapacityReservations = s.decision.capacityReservationView
		v.Capabilities.RequestSpill = s.decision.requestSpill
	}
	if s.decision != nil && s.decision.events != nil {
		v.Capabilities.LifecycleEvents = true
		_, v.Capabilities.TransferEvents = s.KVCache.(DecisionEventSource)
	}
	var queries []DecisionKVQuery
	view := func(r *Request) DecisionRequest {
		spillTokens := int64(0)
		if v.Capabilities.RequestSpill && r.State == StateRunning && !r.TTFTSet && r.EmittedTokens() == 0 && r.ProgressIndex > 0 && r.ProgressIndex < r.InputLen() {
			spillTokens = r.ProgressIndex
		}
		waitUntil := int64(0)
		if s.decision != nil {
			waitUntil = s.decision.waits[r.ID].until
		}
		queries = append(queries, DecisionKVQuery{ID: r.ID, SpillComputedTokens: spillTokens, Input: append([]TokenID(nil), r.PrefillTokens()...),
			CapacityReservation: v.Capabilities.CapacityReservations, InputTokens: r.InputLen(), ClientOutputLimit: int64(r.MaxOutputLen)})
		return DecisionRequest{RecomputeUntilTokens: r.recomputeUntil, ID: r.ID, State: r.State, ArrivalUS: r.ArrivalTime, InputTokens: r.InputLen(), ComputedTokens: r.ProgressIndex, EmittedTokens: r.EmittedTokens(),
			ClientOutputLimit: r.MaxOutputLen, TTFTTargetUS: r.SLOTargetUs, DeadlineUS: r.Deadline, Priority: r.Priority, WaitUntilUS: waitUntil}
	}
	for _, r := range s.WaitQ.Items() {
		v.Waiting = append(v.Waiting, view(r))
	}
	if s.RunningBatch != nil {
		for _, r := range s.RunningBatch.Requests {
			v.Running = append(v.Running, view(r))
		}
	}
	if provider, ok := s.KVCache.(DecisionStateProvider); ok {
		if s.decision != nil && s.decision.prefixSharing {
			v.KV = s.KVCache.(DecisionSharingStateProvider).DecisionStateWithSharing(queries)
		} else {
			v.KV = provider.DecisionState(queries)
		}
	}
	if s.decision != nil && s.decision.prefixProducers {
		v.Capabilities.PrefixProducers = true
		v.PrefixProducers = decisionPrefixProducers(v.Waiting, queries, v.BlockTokens)
	}
	s.markDecisionPrefillPreemptible(&v)
	s.markDecisionPrefillWaitable(&v)
	s.markDecisionPrefillDeferrable(&v)
	return v
}
func cloneDecisionView(v DecisionView) DecisionView {
	v.PrefixProducers = append([]DecisionPrefixProducer(nil), v.PrefixProducers...)
	if v.Estimates != nil {
		estimates := cloneDecisionEstimates(*v.Estimates)
		v.Estimates = &estimates
	}
	v.Waiting = append([]DecisionRequest(nil), v.Waiting...)
	v.Running = append([]DecisionRequest(nil), v.Running...)
	v.KV.Requests = append([]DecisionKVRequest(nil), v.KV.Requests...)
	for i := range v.KV.Requests {
		v.KV.Requests[i].SpillTargets = append([]DecisionSpillTarget(nil), v.KV.Requests[i].SpillTargets...)
		v.KV.Requests[i].PrefixBlocks = append([]DecisionPrefixBlock(nil), v.KV.Requests[i].PrefixBlocks...)
		if c := v.KV.Requests[i].CapacityReservation; c != nil {
			copy := *c
			v.KV.Requests[i].CapacityReservation = &copy
		}
	}
	v.KV.Pools = append([]DecisionPool(nil), v.KV.Pools...)
	v.KV.PendingTransfers = append([]DecisionTransfer(nil), v.KV.PendingTransfers...)
	return v
}
func cloneDecisionPlan(p DecisionPlan) DecisionPlan {
	p.PreemptionStorage = append([]DecisionPreemptionStorage(nil), p.PreemptionStorage...)
	if p.CapacityVictims != nil {
		copy := &DecisionCapacityVictims{Order: append([]string{}, p.CapacityVictims.Order...)}
		for _, a := range p.CapacityVictims.Admission {
			copy.Admission = append(copy.Admission, DecisionCapacityAdmission{Request: a.Request, Victims: append([]string{}, a.Victims...)})
		}
		p.CapacityVictims = copy
	}
	p.Preemptions = append([]string(nil), p.Preemptions...)
	p.PrefillDeferrals = append([]string(nil), p.PrefillDeferrals...)
	p.Promotions = append([]DecisionPromotion(nil), p.Promotions...)
	if p.Admission != nil {
		admission := &DecisionAdmission{}
		if p.Admission.Requests != nil {
			admission.Requests = append([]string{}, p.Admission.Requests...)
		}
		p.Admission = admission
	}
	p.Restores = append([]DecisionRestore(nil), p.Restores...)
	if p.QueueOrder != nil {
		p.QueueOrder = append([]string{}, p.QueueOrder...)
	}
	p.TokenCaps = append([]DecisionTokenCap(nil), p.TokenCaps...)
	p.Waits = append([]DecisionWait(nil), p.Waits...)
	if p.RevisitAtUS != nil {
		at := *p.RevisitAtUS
		p.RevisitAtUS = &at
	}
	return p
}
func cloneDecisionFeedback(f DecisionFeedback) DecisionFeedback {
	f.PreemptionStorage = append([]RequestSpillOutcome(nil), f.PreemptionStorage...)
	f.Capacity = append([]DecisionCapacityOutcome(nil), f.Capacity...)
	f.Preemptions = append([]DecisionPreemptionOutcome(nil), f.Preemptions...)
	f.Promotions = append([]DecisionPromotionOutcome(nil), f.Promotions...)
	f.RestoreUpdates = append([]DecisionRestore(nil), f.RestoreUpdates...)
	f.Grants = append([]DecisionGrant(nil), f.Grants...)
	f.Unselected = append([]DecisionUnselected(nil), f.Unselected...)
	f.WaitUpdates = append([]DecisionWaitOutcome(nil), f.WaitUpdates...)
	if f.Revisit != nil {
		revisit := *f.Revisit
		f.Revisit = &revisit
	}
	return f
}

// ValidateDecision checks the entire plan before any queue/resource mutation.
func ValidateDecision(v DecisionView, p DecisionPlan) error {
	if err := validateDecisionPrefillDeferrals(v, p); err != nil {
		return err
	}
	if err := validateDecisionCapacity(v, p); err != nil {
		return err
	}
	if err := validateDecisionPreemptions(v, p); err != nil {
		return err
	}
	if err := validateDecisionPreemptionStorage(v, p); err != nil {
		return err
	}
	if p.BatchTokenCap < 0 || p.BatchTokenCap > 0 && (!v.Capabilities.BatchTokenCap || p.BatchTokenCap > v.MaxBatchTokens || p.BatchTokenCap < minimumRunningGrants(v, p)) {
		return fmt.Errorf("invalid or unsupported batch token cap")
	}
	if p.Version != v.Version {
		return fmt.Errorf("stale decision version %d, want %d", p.Version, v.Version)
	}
	if err := validateDecisionPromotions(v, p); err != nil {
		return err
	}
	waiting, visible, runningWaitable := map[string]bool{}, map[string]bool{}, map[string]bool{}
	inputBlocks := map[string]int64{}
	for _, r := range v.Waiting {
		if v.BlockTokens > 0 {
			inputBlocks[r.ID] = r.InputTokens / v.BlockTokens
		}
		waiting[r.ID] = true
		visible[r.ID] = true
	}
	for _, r := range v.Running {
		visible[r.ID] = true
		runningWaitable[r.ID] = v.Capabilities.PrefillWaits && r.PrefillWaitable && r.State == StateRunning && r.ComputedTokens > 0 && r.ComputedTokens < r.InputTokens && r.EmittedTokens == 0
	}
	if p.Admission != nil {
		if !v.Capabilities.AdmissionSelection {
			return fmt.Errorf("waiting admission selection is unsupported")
		}
		seen := map[string]bool{}
		for _, id := range p.Admission.Requests {
			if !waiting[id] || seen[id] {
				return fmt.Errorf("unknown or repeated admission request %q", id)
			}
			seen[id] = true
		}
	}
	if p.QueueOrder != nil {
		if !v.Capabilities.QueueOrder || len(p.QueueOrder) != len(waiting) {
			return fmt.Errorf("queue order must cover the current waiting queue")
		}
		seen := map[string]bool{}
		for _, id := range p.QueueOrder {
			if !waiting[id] || seen[id] {
				return fmt.Errorf("unknown or repeated waiting request %q", id)
			}
			seen[id] = true
		}
	}
	seen := map[string]bool{}
	for _, restore := range p.Restores {
		if !v.Capabilities.RestoreChoice || v.BlockTokens <= 0 || !waiting[restore.Request] || seen[restore.Request] || restore.MaxPrefixBlocks < 0 || restore.MaxPrefixBlocks > inputBlocks[restore.Request] {
			return fmt.Errorf("invalid restore limit for %q", restore.Request)
		}
		seen[restore.Request] = true
	}
	seen = map[string]bool{}
	for _, cap := range p.TokenCaps {
		if !v.Capabilities.TokenCaps || !visible[cap.Request] || seen[cap.Request] || cap.Tokens <= 0 || cap.Tokens > v.MaxBatchTokens {
			return fmt.Errorf("invalid token cap for %q", cap.Request)
		}
		seen[cap.Request] = true
	}
	seen = map[string]bool{}
	for _, wait := range p.Waits {
		if !v.Capabilities.WaitUntil || !(waiting[wait.Request] || runningWaitable[wait.Request]) || seen[wait.Request] || wait.UntilUS < 0 || (wait.UntilUS > 0 && wait.UntilUS <= v.NowUS) {
			return fmt.Errorf("invalid wait deadline for %q", wait.Request)
		}
		seen[wait.Request] = true
	}
	if p.RevisitAtUS != nil && (!v.Capabilities.Revisit || *p.RevisitAtUS < 0 || (*p.RevisitAtUS > 0 && (*p.RevisitAtUS <= v.NowUS || len(visible) == 0))) {
		return fmt.Errorf("invalid revisit deadline")
	}
	return nil
}

func (s *Simulator) prepareDecision(now int64) map[string]int64 {
	return s.prepareDecisionKind(now, false)
}

func (s *Simulator) prepareDecisionKind(now int64, control bool) map[string]int64 {
	if s.decision == nil {
		return nil
	}
	d := s.decision
	// Consume only completion obligations visible at this service's start.
	// New completions while service is busy request a later iteration.
	d.controlPending = false
	runtimeStart := time.Now()
	start := time.Now()
	view := s.decisionView(now)
	if d.estimator != nil {
		estimates, err := d.estimator.Estimate(cloneDecisionView(view))
		if err == nil {
			err = ValidateDecisionEstimates(view, estimates)
		}
		if err != nil {
			d.record = DecisionRecord{ControlStep: control, View: view, Plan: DecisionPlan{Version: view.Version}, CostCoverage: "host_measured_sim_time_uncalibrated"}
			s.deliverDecision(DecisionFeedback{Version: view.Version, Status: "rejected", Reason: err.Error()})
			panic(err)
		}
		estimates = cloneDecisionEstimates(estimates)
		view.Estimates = &estimates
		view.Capabilities.CostEstimates = true
	}
	snapshotNS := time.Since(start).Nanoseconds()
	start = time.Now()
	plan := cloneDecisionPlan(d.policy.Decide(cloneDecisionView(view)))
	decisionNS := time.Since(start).Nanoseconds()
	d.record = DecisionRecord{ControlStep: control, View: view, Plan: plan, SnapshotWallNS: snapshotNS, DecisionWallNS: decisionNS, CostCoverage: "host_measured_sim_time_uncalibrated"}
	err := ValidateDecision(view, plan)
	if err == nil {
		err = s.validateDecisionRestores(plan)
	}
	if err != nil {
		d.record.RuntimeWallNS = time.Since(runtimeStart).Nanoseconds()
		s.deliverDecision(DecisionFeedback{Version: view.Version, Status: "rejected", Reason: err.Error()})
		panic(err) // malformed policy is a run failure, never a silent fallback
	}
	if d.cost != nil {
		cost, err := d.cost.Estimate(cloneDecisionView(view), cloneDecisionPlan(plan))
		if err == nil {
			err = cost.validate(now)
		}
		if err != nil {
			d.record.RuntimeWallNS = time.Since(runtimeStart).Nanoseconds()
			s.deliverDecision(DecisionFeedback{Version: view.Version, Status: "rejected", Reason: err.Error()})
			panic(err)
		}
		d.record.Service = &DecisionService{StartUS: now, EndUS: now + cost.ExtraUS, ExtraUS: cost.ExtraUS, Provenance: cost.Provenance}
		d.record.CostCoverage = ExplicitDecisionCostCoverage
		if cost.ExtraUS > 0 {
			ev := &DecisionCompleteEvent{At: now + cost.ExtraUS, waiting: append([]*Request(nil), s.WaitQ.Items()...)}
			if s.RunningBatch != nil {
				ev.running = append([]*Request(nil), s.RunningBatch.Requests...)
			}
			d.pending = ev
			s.stepEvent = ev
			s.Schedule(ev)
			d.record.RuntimeWallNS = time.Since(runtimeStart).Nanoseconds()
			return nil
		}
	}
	d.record.RuntimeWallNS = time.Since(runtimeStart).Nanoseconds()
	return s.applyDecisionPlan()
}

func (s *Simulator) applyDecisionPlan() map[string]int64 {
	d := s.decision
	start := time.Now()
	plan := d.record.Plan
	if len(plan.PrefillDeferrals) > 0 {
		if err := validateDecisionPrefillDeferrals(s.decisionView(s.decisionAppliedAt()), plan); err != nil {
			s.deliverDecision(DecisionFeedback{Version: d.record.View.Version, Status: "rejected", Reason: err.Error()})
			panic(err)
		}
	}
	if len(plan.Preemptions) > 0 || plan.CapacityVictims != nil {
		if err := validateDecisionPreemptions(s.decisionView(s.decisionAppliedAt()), plan); err != nil {
			s.deliverDecision(DecisionFeedback{Version: d.record.View.Version, Status: "rejected", Reason: err.Error()})
			panic(err)
		}
		if err := validateDecisionCapacity(s.decisionView(s.decisionAppliedAt()), plan); err != nil {
			s.deliverDecision(DecisionFeedback{Version: d.record.View.Version, Status: "rejected", Reason: err.Error()})
			panic(err)
		}
	}
	// Service time may have advanced storage. Recheck before any wait, queue,
	// or restore metadata changes; a malformed plan has no partial effects.
	if err := s.validateDecisionRestores(plan); err != nil {
		s.deliverDecision(DecisionFeedback{Version: d.record.View.Version, Status: "rejected", Reason: err.Error()})
		panic(err)
	}
	if len(plan.Restores) > 0 {
		s.KVCache.(RestoreDecisionStore).ApplyRestoreDecisions(plan.Restores)
		d.record.Feedback.RestoreUpdates = append([]DecisionRestore(nil), plan.Restores...)
	}
	s.applyDecisionPromotions(plan)
	s.applyDecisionWaits(s.decisionAppliedAt())
	d.batchTokenCap = plan.BatchTokenCap
	d.deferredPrefills = nil
	if len(plan.PrefillDeferrals) > 0 {
		d.deferredPrefills = map[string]bool{}
		for _, id := range plan.PrefillDeferrals {
			d.deferredPrefills[id] = true
		}
	}
	d.admission = nil
	if plan.Admission != nil {
		d.admission = map[string]bool{}
		for _, id := range plan.Admission.Requests {
			d.admission[id] = true
		}
	}
	if plan.QueueOrder != nil {
		rank := map[string]int{}
		for i, id := range plan.QueueOrder {
			rank[id] = i
		}
		s.WaitQ.Reorder(func(reqs []*Request) {
			sort.SliceStable(reqs, func(i, j int) bool { return rank[reqs[i].ID] < rank[reqs[j].ID] })
		})
	}
	d.prefillPreemptions = nil
	d.preemptionStorage = append([]DecisionPreemptionStorage(nil), plan.PreemptionStorage...)
	d.capacityVictims = cloneDecisionPlan(plan).CapacityVictims
	for _, id := range plan.Preemptions {
		for _, req := range s.RunningBatch.Requests {
			if req.ID == id {
				d.prefillPreemptions = append(d.prefillPreemptions, req)
				break
			}
		}
	}
	caps := map[string]int64{}
	for _, cap := range plan.TokenCaps {
		caps[cap.Request] = cap.Tokens
	}
	d.record.RuntimeWallNS += time.Since(start).Nanoseconds()
	return caps
}
func (s *Simulator) finishDecision(result BatchResult) {
	if s.decision == nil {
		return
	}
	d := s.decision
	d.record.ExecutionWork = result.ExecutionWork
	runtimeStart := time.Now()
	f := DecisionFeedback{Version: d.record.View.Version, Status: "applied", WaitUpdates: d.record.Feedback.WaitUpdates, Revisit: d.record.Feedback.Revisit}
	f.RestoreUpdates = d.record.Feedback.RestoreUpdates
	f.Capacity = append([]DecisionCapacityOutcome(nil), result.Capacity...)
	f.PreemptionStorage = append([]RequestSpillOutcome(nil), result.PreemptionStorage...)
	f.Promotions = d.record.Feedback.Promotions
	selected, preempted, pending := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, r := range result.RunningBatch.Requests {
		if r.NumNewTokens > 0 {
			f.Grants = append(f.Grants, DecisionGrant{Request: r.ID, Tokens: int64(r.NumNewTokens)})
			selected[r.ID] = true
		}
	}
	for _, r := range result.Preempted {
		preempted[r.Request.ID] = true
		if r.Reason == "policy_prefill" || r.Reason == "policy_capacity_prefill" {
			f.Preemptions = append(f.Preemptions, DecisionPreemptionOutcome{Request: r.Request.ID, ComputedTokensBefore: r.ComputedTokensBefore, Status: "requeued"})
		}
	}
	for _, r := range s.decisionView(s.decisionAppliedAt()).KV.Requests {
		pending[r.ID] = r.TransferPending || r.Deferred
	}
	for _, r := range append(append([]DecisionRequest(nil), d.record.View.Running...), d.record.View.Waiting...) {
		if selected[r.ID] {
			continue
		}
		reason := "runtime_not_selected" // no invented capacity/dependency cause
		if d.deferredPrefills[r.ID] {
			reason = "policy_prefill_deferred"
		}
		if pending[r.ID] {
			reason = "kv_pending"
		}
		if s.decisionWaitsUntil(r.ID) > s.decisionAppliedAt() {
			reason = "policy_wait"
		}
		if d.admission != nil && !d.admission[r.ID] {
			for _, waiting := range d.record.View.Waiting {
				if waiting.ID == r.ID {
					reason = "policy_not_admitted"
					break
				}
			}
		}
		if preempted[r.ID] {
			reason = "preempted"
		}
		f.Unselected = append(f.Unselected, DecisionUnselected{Request: r.ID, Reason: reason})
	}
	d.record.RuntimeWallNS += time.Since(runtimeStart).Nanoseconds()
	s.deliverDecision(f)
}
func (s *Simulator) deliverDecision(f DecisionFeedback) {
	if s.deferExecutionFeedback(f) {
		return
	}
	s.deliverDecisionAt(f, s.decisionAppliedAt())
}

func (s *Simulator) deliverDecisionAt(f DecisionFeedback, now int64) {
	d := s.decision
	f.AtUS = now
	d.record.Feedback = f
	start := time.Now()
	d.policy.Observe(cloneDecisionFeedback(f))
	d.record.FeedbackWallNS = time.Since(start).Nanoseconds()
	d.record.RuntimeWallNS += d.record.FeedbackWallNS
	if d.observe != nil {
		d.observe(d.record)
	}
}

func (s *Simulator) decisionAppliedAt() int64 {
	if service := s.decision.record.Service; service != nil {
		return service.EndUS
	}
	return s.decision.record.View.NowUS
}

func (s *Simulator) validateDecisionRestores(plan DecisionPlan) error {
	if len(plan.Restores) == 0 {
		return nil
	}
	backend, ok := s.KVCache.(RestoreDecisionStore)
	if !ok || !backend.RestoreDecisionsEnabled() {
		return fmt.Errorf("restore decisions unsupported by this backend")
	}
	return backend.ValidateRestoreDecisions(plan.Restores)
}
