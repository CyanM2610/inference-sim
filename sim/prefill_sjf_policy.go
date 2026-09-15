package sim

// PrefillSJFPolicy is an input-length preemptive example, not Cascade or an
// output-length oracle. With every sequence slot occupied, one shorter waiting
// request can replace the longest eligible running prefill for this step.
// It deliberately does not estimate migration/recomputation cost or promise SLOs.
type PrefillSJFPolicy struct{}

func (*PrefillSJFPolicy) Decide(v DecisionView) DecisionPlan {
	if !v.Capabilities.PrefillPreemption {
		panic("prefill_sjf requires prefill_preemption")
	}
	base, _ := NewQueueDecisionPolicy("sjf", 0)
	plan := base.Decide(v)
	if len(v.Waiting) == 0 || int64(len(v.Running)) < v.MaxSequences {
		return plan
	}
	blocked := map[string]bool{}
	for _, r := range v.KV.Requests {
		blocked[r.ID] = r.TransferPending || r.Deferred
	}
	var shortest *DecisionRequest
	for i := range v.Waiting {
		r := &v.Waiting[i]
		if r.WaitUntilUS > v.NowUS || blocked[r.ID] {
			continue
		}
		if shortest == nil || r.InputTokens < shortest.InputTokens || r.InputTokens == shortest.InputTokens && (r.ArrivalUS < shortest.ArrivalUS || r.ArrivalUS == shortest.ArrivalUS && r.ID < shortest.ID) {
			shortest = r
		}
	}
	if shortest == nil {
		return plan
	}
	var victim *DecisionRequest
	for i := range v.Running {
		r := &v.Running[i]
		if !r.PrefillPreemptible || r.InputTokens <= shortest.InputTokens {
			continue
		}
		if victim == nil || r.InputTokens > victim.InputTokens || r.InputTokens == victim.InputTokens && r.ID > victim.ID {
			victim = r
		}
	}
	if victim != nil {
		plan.Preemptions = []string{victim.ID}
		plan.Admission = &DecisionAdmission{Requests: []string{shortest.ID}}
	}
	return plan
}

func (*PrefillSJFPolicy) Observe(DecisionFeedback) {}
