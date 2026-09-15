// kvdecision evaluates detached snapshots for cross-language contract checks.
// It is an offline reference, not part of a measured inference execution path.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/policylab"
)

type input struct {
	// Replay preserves decision/actual-feedback ordering for stateful policies.
	Replay []decisionReplay `json:"replay,omitempty"`
	// Feedback is applied only after evaluating the current view and plan.
	Feedback *sim.DecisionFeedback `json:"feedback,omitempty"`
	// ViewHistory replays prior decisions on the same fresh policy instance.
	// History is the older feedback-only input; combining the two cannot specify
	// feedback/decision ordering and is therefore rejected.
	ViewHistory      []sim.DecisionView                `json:"view_history,omitempty"`
	History          []sim.DecisionFeedback            `json:"history,omitempty"`
	ProducerWait     *policylab.ProducerWaitConfig     `json:"producer_wait,omitempty"`
	BalancedBatch    *policylab.BalancedBatchConfig    `json:"balanced_batch,omitempty"`
	PrefillWaitProbe *policylab.PrefillWaitProbeConfig `json:"prefill_wait_probe,omitempty"`
	Config           policylab.Config                  `json:"config"`
	View             sim.DecisionView                  `json:"view"`
	Mode             string                            `json:"mode"`
	Guardband        float64                           `json:"guardband"`
	PrefillCap       int64                             `json:"prefill_cap"`
	IncludeWork      bool                              `json:"include_work,omitempty"`
}

type decisionReplay struct {
	View     sim.DecisionView     `json:"view"`
	Feedback sim.DecisionFeedback `json:"feedback"`
}

type budgetPolicy interface {
	sim.DecisionPolicy
	RequestBudgets(int64) []sim.RequestBudget
}

type output struct {
	PolicyState               *sim.DecodeFillState    `json:"policy_state,omitempty"`
	QueueServiceAfterFeedback bool                    `json:"queue_service_after_feedback,omitempty"`
	QueueService              *sim.BudgetQueueService `json:"queue_service,omitempty"`
	Estimates                 *sim.DecisionEstimates  `json:"estimates,omitempty"`
	Plan                      sim.DecisionPlan        `json:"plan"`
	Work                      map[string]int64        `json:"work,omitempty"`
	Budgets                   []sim.RequestBudget     `json:"budgets,omitempty"`
}

func evaluate(in input) (output, error) {
	var out output
	balanced := in.BalancedBatch
	if c := in.Config.DecisionPolicy; c != nil && c.BalancedBatch != nil && c.BalancedBatch.DecodeFill {
		if balanced != nil && *balanced != *c.BalancedBatch {
			return out, fmt.Errorf("conflicting decode-fill configuration")
		}
		balanced = c.BalancedBatch
	}
	if balanced != nil && balanced.DecodeFill && (len(in.ViewHistory) > 0 || len(in.History) > 0) {
		return out, fmt.Errorf("decode fill requires ordered view/actual-feedback replay")
	}
	var decodeFill *sim.DecodeFillPolicy
	var deferralProbe *policylab.PrefillDeferralProbeConfig
	if in.Config.DecisionPolicy != nil {
		deferralProbe = in.Config.DecisionPolicy.PrefillDeferralProbe
	}
	if deferralProbe != nil && (len(in.ViewHistory) > 0 || len(in.History) > 0) {
		return out, fmt.Errorf("prefill_deferral_probe requires ordered view/actual-feedback replay")
	}
	probe := in.PrefillWaitProbe
	if in.Config.DecisionPolicy != nil && in.Config.DecisionPolicy.PrefillWaitProbe != nil {
		if probe != nil {
			return out, fmt.Errorf("choose one prefill_wait_probe configuration")
		}
		probe = in.Config.DecisionPolicy.PrefillWaitProbe
	}
	retained := in.Mode == "initial_budget" || in.Mode == "budget_pressure" || in.Mode == "budget_pressure_fcfs" || in.Mode == "budget_queues"
	if c := in.Config.DecisionPolicy; c != nil && c.BudgetRestore != nil && c.BudgetRestore.ReestimateOnDecodeDrop && in.Mode != "budget_queues" {
		return out, fmt.Errorf("decode-drop budget reestimate requires budget_queues mode")
	}
	if len(in.Replay) > 0 && (len(in.ViewHistory) > 0 || len(in.History) > 0) {
		return out, fmt.Errorf("replay cannot be combined with legacy histories")
	}
	if in.Mode == "budget_queues" && (len(in.ViewHistory) > 0 || len(in.History) > 0) {
		return out, fmt.Errorf("budget_queues requires replay pairs with actual feedback, not view-only or feedback-only history")
	}
	if in.Mode == "budget_queues" && in.IncludeWork {
		return out, fmt.Errorf("budget queue service work has no independent cost model")
	}
	if len(in.ViewHistory) > 0 && !retained && probe == nil {
		return out, fmt.Errorf("view_history requires a retained-budget policy or prefill_wait_probe")
	}
	if len(in.ViewHistory) > 0 && len(in.History) > 0 {
		return out, fmt.Errorf("view_history and history cannot specify interleaved feedback ordering")
	}
	var policy sim.DecisionPolicy
	var budget budgetPolicy
	var queues *sim.BudgetQueuePolicy
	var estimator sim.DecisionEstimator
	var err error
	switch {
	case in.Mode == "prefetch_fcfs" || in.Mode == "demand_fcfs":
		policy, err = sim.NewPrefixQueuePolicy("fcfs", in.Mode == "prefetch_fcfs")
	case in.Mode == "prefill_sjf":
		policy = &sim.PrefillSJFPolicy{}
	case in.Mode == "sjf":
		policy, err = sim.NewQueueDecisionPolicy("sjf", 0)
	case balanced != nil:
		c := balanced
		var base *sim.BalancedBatchPolicy
		base, err = sim.NewBalancedBatchPolicyWithOptions(c.MaxLoadComputeRatio, c.TokenBudget, c.MaxRequests, in.PrefillCap,
			sim.BalancedBatchOptions{BundleHits: c.BundleHits, ReadyComputeBudget: c.ReadyComputeBudget})
		policy = base
		if err == nil && c.DecodeFill {
			if in.ProducerWait != nil {
				return out, fmt.Errorf("decode fill cannot be combined with producer wait")
			}
			decodeFill, err = sim.NewDecodeFillPolicy(base)
			policy = decodeFill
		}
		if err == nil && in.ProducerWait != nil {
			w := in.ProducerWait
			policy, err = sim.NewProducerWaitPolicy(policy, w.MinSharedTokens, w.MaxWaitUS)
		}
	default:
		estimator, err = policylab.NewRestoreDecisionEstimator(in.Config)
		if err != nil {
			return out, err
		}
		switch in.Mode {
		case "budget_queues":
			primary, best := int64(4), int64(1)
			options := sim.BudgetQueueOptions{}
			if c := in.Config.DecisionPolicy; c != nil && c.BudgetRestore != nil {
				options.ReestimateOnDecodeDrop = c.BudgetRestore.ReestimateOnDecodeDrop
				if c.BudgetRestore.PrimaryWeight != 0 || c.BudgetRestore.BestEffortWeight != 0 {
					primary, best = c.BudgetRestore.PrimaryWeight, c.BudgetRestore.BestEffortWeight
				}
			}
			queues, err = sim.NewBudgetQueuePolicyWithOptions(in.Guardband, in.PrefillCap, primary, best, options)
			budget = queues
		case "restore_bound":
			budget, err = sim.NewBudgetRestorePolicy(in.Guardband, in.PrefillCap)
		case "completion_bound":
			budget, err = sim.NewCompletionBudgetRestorePolicy(in.Guardband, in.PrefillCap)
		case "initial_budget":
			budget, err = sim.NewInitialBudgetRestorePolicy(in.Guardband, in.PrefillCap)
		case "budget_pressure":
			budget, err = sim.NewBudgetPressurePolicy(in.Guardband, in.PrefillCap)
		case "budget_pressure_fcfs":
			budget, err = sim.NewBudgetPressureFIFOPolicy(in.Guardband, in.PrefillCap)
		default:
			return out, fmt.Errorf("unknown budget mode %q", in.Mode)
		}
		policy = budget
	}
	if err != nil {
		return out, err
	}
	if c := in.Config.DecisionPolicy; c != nil && c.PreemptionStorage != nil {
		policy, err = c.PreemptionStorage.NewPolicy(policy)
		if err != nil {
			return out, err
		}
	}
	if probe != nil {
		policy, err = probe.NewPolicy(policy)
		if err != nil {
			return out, err
		}
	}
	for _, feedback := range in.History {
		policy.Observe(feedback)
	}
	if deferralProbe != nil {
		policy, err = deferralProbe.NewPolicy(policy)
		if err != nil {
			return out, err
		}
	}
	decide := func(view sim.DecisionView) (sim.DecisionPlan, *sim.DecisionEstimates, error) {
		if decodeFill != nil && (!view.Capabilities.PrefillDeferrals || !view.Capabilities.Promotions || view.Capabilities.PromotionRetention != "request" || !view.KV.TransfersKnown || view.BlockTokens <= 0) {
			return sim.DecisionPlan{}, nil, fmt.Errorf("decode fill lacks declared runtime capabilities")
		}
		if deferralProbe != nil && !view.Capabilities.PrefillDeferrals {
			return sim.DecisionPlan{}, nil, fmt.Errorf("deferral probe requires per-round prefill capability")
		}
		if probe != nil && !view.Capabilities.PrefillWaits {
			return sim.DecisionPlan{}, nil, fmt.Errorf("prefill_wait_probe requires resident wait capability")
		}
		var estimates *sim.DecisionEstimates
		if estimator != nil {
			value, err := estimator.Estimate(view)
			if err != nil {
				return sim.DecisionPlan{}, nil, err
			}
			if err := sim.ValidateDecisionEstimates(view, value); err != nil {
				return sim.DecisionPlan{}, nil, err
			}
			estimates = &value
			view.Estimates = estimates
		}
		plan := policy.Decide(view)
		return plan, estimates, sim.ValidateDecision(view, plan)
	}
	for i, previous := range in.ViewHistory {
		if _, _, err := decide(previous); err != nil {
			return out, fmt.Errorf("view_history[%d]: %w", i, err)
		}
	}
	var lastVersion uint64
	var lastFeedbackUS int64
	for i, previous := range in.Replay {
		if previous.View.Version >= in.View.Version || i > 0 && (previous.View.Version <= lastVersion || previous.View.NowUS < lastFeedbackUS) ||
			previous.View.NowUS > in.View.NowUS || previous.Feedback.AtUS > in.View.NowUS {
			return out, fmt.Errorf("replay[%d] is not ordered prior history", i)
		}
		lastVersion = previous.View.Version
		lastFeedbackUS = previous.Feedback.AtUS
		plan, _, e := decide(previous.View)
		if e != nil {
			return out, fmt.Errorf("replay[%d]: %w", i, e)
		}
		if e := validateReplayFeedback(previous.View, plan, previous.Feedback); e != nil {
			return out, fmt.Errorf("replay[%d]: %w", i, e)
		}
		policy.Observe(previous.Feedback)
	}
	out.Plan, out.Estimates, err = decide(in.View)
	if err != nil {
		return out, err
	}
	if in.Feedback != nil {
		if err := validateReplayFeedback(in.View, out.Plan, *in.Feedback); err != nil {
			return out, err
		}
		policy.Observe(*in.Feedback)
	}
	if decodeFill != nil {
		state := decodeFill.State()
		out.PolicyState = &state
	}
	if budget != nil {
		out.Budgets = budget.RequestBudgets(in.View.NowUS)
		if in.IncludeWork {
			in.View.Estimates = out.Estimates
			out.Work, err = policylab.RestoreDecisionWork(in.Config, in.View)
		}
	}
	if queues != nil {
		state := queues.ServiceState()
		out.QueueService = &state
		out.QueueServiceAfterFeedback = in.Feedback != nil
	}
	return out, err
}

func validateReplayFeedback(v sim.DecisionView, plan sim.DecisionPlan, f sim.DecisionFeedback) error {
	if f.Version != v.Version || f.AtUS < v.NowUS {
		return fmt.Errorf("feedback version/time does not match its view")
	}
	if f.Status != "applied" {
		if (f.Status != "superseded" && f.Status != "rejected") || len(f.Grants) != 0 {
			return fmt.Errorf("unapplied or unknown feedback contains invalid work")
		}
		return nil
	}
	visible, waiting, admitted, removed := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, r := range v.Waiting {
		visible[r.ID], waiting[r.ID] = true, true
	}
	for _, r := range v.Running {
		visible[r.ID] = true
	}
	for _, id := range plan.Preemptions {
		removed[id] = true
	}
	for _, id := range plan.PrefillDeferrals {
		removed[id] = true
	}
	if plan.Admission != nil {
		for _, id := range plan.Admission.Requests {
			admitted[id] = true
		}
	}
	caps := map[string]int64{}
	for _, cap := range plan.TokenCaps {
		caps[cap.Request] = cap.Tokens
	}
	remaining := v.MaxBatchTokens
	if plan.BatchTokenCap > 0 {
		remaining = min(remaining, plan.BatchTokenCap)
	}
	seen := map[string]bool{}
	for _, g := range f.Grants {
		if !visible[g.Request] || seen[g.Request] || removed[g.Request] || g.Tokens <= 0 || g.Tokens > remaining ||
			caps[g.Request] > 0 && g.Tokens > caps[g.Request] || waiting[g.Request] && plan.Admission != nil && !admitted[g.Request] {
			return fmt.Errorf("feedback grants violate the replayed plan")
		}
		seen[g.Request] = true
		remaining -= g.Tokens
	}
	if len(plan.PrefillDeferrals) > 0 {
		unselected := map[string]bool{}
		for _, r := range f.Unselected {
			if !visible[r.Request] || seen[r.Request] || unselected[r.Request] {
				return fmt.Errorf("invalid unselected feedback for deferral replay")
			}
			unselected[r.Request] = true
		}
		for _, id := range plan.PrefillDeferrals {
			if !unselected[id] {
				return fmt.Errorf("deferral replay lacks actual unselected feedback")
			}
		}
	}
	if len(plan.Promotions) > 0 {
		endpoints := map[string]int64{}
		for _, a := range plan.Promotions {
			endpoints[a.Request] = a.MaxPrefixBlocks
		}
		for _, o := range f.Promotions {
			end, ok := endpoints[o.Request]
			if !ok || o.MaxPrefixBlocks != end || o.ReadyBlocks < 0 || o.PendingBlocks < 0 || o.StartedBlocks < 0 || o.BlockedBlocks < 0 ||
				o.ReadyBlocks+o.PendingBlocks+o.StartedBlocks+o.BlockedBlocks != end {
				return fmt.Errorf("promotion feedback does not partition its actual plan")
			}
			delete(endpoints, o.Request)
		}
		if len(endpoints) > 0 {
			return fmt.Errorf("promotion replay lacks actual outcome")
		}
	}
	return nil
}

func runStreams(reader io.Reader, writer io.Writer) error {
	decoder, encoder := json.NewDecoder(reader), json.NewEncoder(writer)
	decoder.DisallowUnknownFields()
	for {
		var in input
		if err := decoder.Decode(&in); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		out, err := evaluate(in)
		if err != nil {
			return err
		}
		if err := encoder.Encode(out); err != nil {
			return err
		}
	}
}

func main() {
	if err := runStreams(os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
