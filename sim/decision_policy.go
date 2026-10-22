package sim

import (
	"fmt"
	"sort"
	"time"
)

// DecisionRequest is a detached view of an arrived request. ClientOutputLimit
// is an explicit client budget; actual future output tokens/length are absent.
type DecisionRequest struct {
	ID                string       `json:"id"`
	State             RequestState `json:"state"`
	ArrivalUS         int64        `json:"arrival_us"`
	InputTokens       int64        `json:"input_tokens"`
	ComputedTokens    int64        `json:"computed_tokens"`
	EmittedTokens     int64        `json:"emitted_tokens"`
	ClientOutputLimit int          `json:"client_output_limit"`
	TTFTTargetUS      int64        `json:"ttft_target_us"`
	DeadlineUS        int64        `json:"deadline_us"`
	Priority          float64      `json:"priority"`
}

type DecisionKVRequest struct {
	ID                      string `json:"id"`
	LocalPrefixTokens       int64  `json:"local_prefix_tokens"`
	RecoverablePrefixTokens int64  `json:"recoverable_prefix_tokens"`
	TransferPending         bool   `json:"transfer_pending"`
	Deferred                bool   `json:"deferred"`
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
	PrefixStateKnown bool                `json:"prefix_state_known"`
	TransfersKnown   bool                `json:"transfers_known"`
	Requests         []DecisionKVRequest `json:"requests"`
	Pools            []DecisionPool      `json:"pools"`
	PendingTransfers []DecisionTransfer  `json:"pending_transfers"`
}

// DecisionKVQuery is for the trusted storage runtime, never sent to a policy.
type DecisionKVQuery struct {
	ID    string
	Input []TokenID
}
type DecisionStateProvider interface {
	DecisionState([]DecisionKVQuery) DecisionKVState
}

type DecisionCapabilities struct {
	QueueOrder         bool `json:"queue_order"`
	TokenCaps          bool `json:"token_caps"`
	ReadyRestoresFirst bool `json:"ready_restores_first"`
	RestoreChoice      bool `json:"restore_choice"`
	Placement          bool `json:"placement"`
	TransferPriority   bool `json:"transfer_priority"`
	CostEstimates      bool `json:"cost_estimates"`
}
type DecisionView struct {
	Version           uint64               `json:"version"`
	NowUS             int64                `json:"now_us"`
	Waiting           []DecisionRequest    `json:"waiting"`
	Running           []DecisionRequest    `json:"running"`
	MaxBatchTokens    int64                `json:"max_batch_tokens"`
	MaxSequences      int64                `json:"max_sequences"`
	PrefillChunk      int64                `json:"prefill_chunk"`
	BlockTokens       int64                `json:"block_tokens"`
	HBMCapacityBlocks int64                `json:"hbm_capacity_blocks"`
	HBMFreeBlocks     int64                `json:"hbm_free_blocks"`
	KV                DecisionKVState      `json:"kv"`
	Capabilities      DecisionCapabilities `json:"capabilities"`
}

type DecisionTokenCap struct {
	Request string `json:"request"`
	Tokens  int64  `json:"tokens"`
}

// QueueOrder is nil for the runtime's current order, or a complete permutation
// of Waiting. TokenCaps are positive upper bounds, not promised token grants.
// Explicit waiting, preemption and restore/placement actions are not yet open.
type DecisionPlan struct {
	Version    uint64             `json:"version"`
	QueueOrder []string           `json:"queue_order,omitempty"`
	TokenCaps  []DecisionTokenCap `json:"token_caps,omitempty"`
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
	Version    uint64               `json:"version"`
	AtUS       int64                `json:"at_us"`
	Status     string               `json:"status"`
	Reason     string               `json:"reason,omitempty"`
	Grants     []DecisionGrant      `json:"grants,omitempty"`
	Unselected []DecisionUnselected `json:"unselected,omitempty"`
}
type DecisionRecord struct {
	Instance       string           `json:"instance,omitempty"`
	View           DecisionView     `json:"view"`
	Plan           DecisionPlan     `json:"plan"`
	Feedback       DecisionFeedback `json:"feedback"`
	SnapshotWallNS int64            `json:"snapshot_wall_ns"`
	DecisionWallNS int64            `json:"decision_wall_ns"`
	FeedbackWallNS int64            `json:"feedback_wall_ns"`
	RuntimeWallNS  int64            `json:"runtime_wall_ns"`
	CostCoverage   string           `json:"cost_coverage"`
	Service        *DecisionService `json:"service,omitempty"`
}
type DecisionPolicy interface {
	Decide(DecisionView) DecisionPlan
	Observe(DecisionFeedback)
}
type decisionRuntime struct {
	policy  DecisionPolicy
	observe func(DecisionRecord)
	record  DecisionRecord
	cost    DecisionCostModel
	pending *DecisionCompleteEvent
}

func (s *Simulator) SetDecisionPolicy(policy DecisionPolicy, observe func(DecisionRecord)) error {
	if policy == nil || s.stepCount != 0 {
		return fmt.Errorf("decision policy must be installed before execution")
	}
	if _, ok := s.batchFormation.(*VLLMBatchFormation); !ok {
		return fmt.Errorf("decision token caps require VLLMBatchFormation")
	}
	s.decision = &decisionRuntime{policy: policy, observe: observe}
	return nil
}

func (s *Simulator) decisionView(now int64) DecisionView {
	v := DecisionView{Version: uint64(s.stepCount), NowUS: now, MaxBatchTokens: s.maxNumBatchedTokens, MaxSequences: s.maxNumSeqs, PrefillChunk: s.longPrefillTokenThreshold,
		BlockTokens: s.KVCache.BlockSize(), HBMCapacityBlocks: s.KVCache.TotalCapacity(), HBMFreeBlocks: s.KVCache.TotalCapacity() - s.KVCache.UsedBlocks(),
		Capabilities: DecisionCapabilities{QueueOrder: true, TokenCaps: true}}
	_, v.Capabilities.ReadyRestoresFirst = s.KVCache.(DeferredAdmissionStore)
	var queries []DecisionKVQuery
	view := func(r *Request) DecisionRequest {
		emitted := int64(0)
		if r.TTFTSet {
			emitted = max(0, r.ProgressIndex-r.InputLen()+1)
		}
		queries = append(queries, DecisionKVQuery{ID: r.ID, Input: append([]TokenID(nil), r.FullInputTokens()...)})
		return DecisionRequest{ID: r.ID, State: r.State, ArrivalUS: r.ArrivalTime, InputTokens: r.InputLen(), ComputedTokens: r.ProgressIndex, EmittedTokens: emitted,
			ClientOutputLimit: r.MaxOutputLen, TTFTTargetUS: r.SLOTargetUs, DeadlineUS: r.Deadline, Priority: r.Priority}
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
		v.KV = provider.DecisionState(queries)
	}
	return v
}
func cloneDecisionView(v DecisionView) DecisionView {
	v.Waiting = append([]DecisionRequest(nil), v.Waiting...)
	v.Running = append([]DecisionRequest(nil), v.Running...)
	v.KV.Requests = append([]DecisionKVRequest(nil), v.KV.Requests...)
	v.KV.Pools = append([]DecisionPool(nil), v.KV.Pools...)
	v.KV.PendingTransfers = append([]DecisionTransfer(nil), v.KV.PendingTransfers...)
	return v
}
func cloneDecisionPlan(p DecisionPlan) DecisionPlan {
	if p.QueueOrder != nil {
		p.QueueOrder = append([]string{}, p.QueueOrder...)
	}
	p.TokenCaps = append([]DecisionTokenCap(nil), p.TokenCaps...)
	return p
}
func cloneDecisionFeedback(f DecisionFeedback) DecisionFeedback {
	f.Grants = append([]DecisionGrant(nil), f.Grants...)
	f.Unselected = append([]DecisionUnselected(nil), f.Unselected...)
	return f
}

// ValidateDecision checks the entire plan before any queue/resource mutation.
func ValidateDecision(v DecisionView, p DecisionPlan) error {
	if p.Version != v.Version {
		return fmt.Errorf("stale decision version %d, want %d", p.Version, v.Version)
	}
	waiting, visible := map[string]bool{}, map[string]bool{}
	for _, r := range v.Waiting {
		waiting[r.ID] = true
		visible[r.ID] = true
	}
	for _, r := range v.Running {
		visible[r.ID] = true
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
	for _, cap := range p.TokenCaps {
		if !v.Capabilities.TokenCaps || !visible[cap.Request] || seen[cap.Request] || cap.Tokens <= 0 || cap.Tokens > v.MaxBatchTokens {
			return fmt.Errorf("invalid token cap for %q", cap.Request)
		}
		seen[cap.Request] = true
	}
	return nil
}

func (s *Simulator) prepareDecision(now int64) map[string]int64 {
	if s.decision == nil {
		return nil
	}
	d := s.decision
	runtimeStart := time.Now()
	start := time.Now()
	view := s.decisionView(now)
	snapshotNS := time.Since(start).Nanoseconds()
	start = time.Now()
	plan := cloneDecisionPlan(d.policy.Decide(cloneDecisionView(view)))
	decisionNS := time.Since(start).Nanoseconds()
	d.record = DecisionRecord{View: view, Plan: plan, SnapshotWallNS: snapshotNS, DecisionWallNS: decisionNS, CostCoverage: "host_measured_sim_time_uncalibrated"}
	if err := ValidateDecision(view, plan); err != nil {
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
	if plan.QueueOrder != nil {
		rank := map[string]int{}
		for i, id := range plan.QueueOrder {
			rank[id] = i
		}
		s.WaitQ.Reorder(func(reqs []*Request) {
			sort.SliceStable(reqs, func(i, j int) bool { return rank[reqs[i].ID] < rank[reqs[j].ID] })
		})
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
	runtimeStart := time.Now()
	f := DecisionFeedback{Version: d.record.View.Version, Status: "applied"}
	selected, preempted, pending := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, r := range result.RunningBatch.Requests {
		if r.NumNewTokens > 0 {
			f.Grants = append(f.Grants, DecisionGrant{Request: r.ID, Tokens: int64(r.NumNewTokens)})
			selected[r.ID] = true
		}
	}
	for _, r := range result.Preempted {
		preempted[r.Request.ID] = true
	}
	for _, r := range s.decisionView(s.decisionAppliedAt()).KV.Requests {
		pending[r.ID] = r.TransferPending || r.Deferred
	}
	for _, r := range append(append([]DecisionRequest(nil), d.record.View.Running...), d.record.View.Waiting...) {
		if selected[r.ID] {
			continue
		}
		reason := "runtime_not_selected" // no invented capacity/dependency cause
		if pending[r.ID] {
			reason = "kv_pending"
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
	d := s.decision
	f.AtUS = s.decisionAppliedAt()
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
