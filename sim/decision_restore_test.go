package sim

import (
	"fmt"
	"testing"
)

type restoreTestStore struct {
	KVStore
	enabled bool
	blocked bool
	updates []DecisionRestore
}

func (s *restoreTestStore) RestoreDecisionsEnabled() bool { return s.enabled }
func (s *restoreTestStore) ValidateRestoreDecisions([]DecisionRestore) error {
	if s.blocked {
		return fmt.Errorf("pending load range cannot change")
	}
	return nil
}
func (s *restoreTestStore) ApplyRestoreDecisions(u []DecisionRestore) {
	s.updates = append(s.updates, u...)
}

func TestDecisionRestoreRejectsBeforeOtherActions(t *testing.T) {
	for _, name := range []string{"unsupported", "future", "negative", "beyond_prompt", "duplicate", "pending"} {
		t.Run(name, func(t *testing.T) {
			s := decisionFixture()
			s.batchCompletionEvents = true
			store := &restoreTestStore{KVStore: s.KVCache, enabled: name != "unsupported", blocked: name == "pending"}
			s.KVCache = store
			p := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan {
				u := DecisionRestore{Request: "a", MaxPrefixBlocks: 1}
				switch name {
				case "future":
					u.Request = "future"
				case "negative":
					u.MaxPrefixBlocks = -1
				case "beyond_prompt":
					u.MaxPrefixBlocks = 3
				}
				plan := DecisionPlan{Version: v.Version, QueueOrder: []string{"b", "a"}, Waits: []DecisionWait{{Request: "b", UntilUS: 100}}, Restores: []DecisionRestore{u}}
				if name == "duplicate" {
					plan.Restores = append(plan.Restores, u)
				}
				return plan
			}}
			if err := s.SetDecisionPolicy(p, nil); err != nil {
				t.Fatal(err)
			}
			func() {
				defer func() {
					if recover() == nil {
						t.Error("invalid plan did not fail")
					}
				}()
				s.prepareDecision(10)
			}()
			if len(store.updates) != 0 || len(s.decision.waits) != 0 || s.WaitQ.Peek().ID != "a" || s.KVCache.UsedBlocks() != 0 {
				t.Fatal("partial mutation before rejection")
			}
			if len(p.feedback) != 1 || p.feedback[0].Status != "rejected" {
				t.Fatal("missing rejection feedback")
			}
		})
	}
}

func TestDecisionRestoreFeedbackAndPlanAreDetached(t *testing.T) {
	s := decisionFixture()
	store := &restoreTestStore{KVStore: s.KVCache, enabled: true}
	s.KVCache = store
	u := []DecisionRestore{{Request: "a", MaxPrefixBlocks: 0}}
	p := &testDecisionPolicy{decide: func(v DecisionView) DecisionPlan { return DecisionPlan{Version: v.Version, Restores: u} }}
	if err := s.SetDecisionPolicy(p, nil); err != nil {
		t.Fatal(err)
	}
	s.prepareDecision(10)
	u[0].MaxPrefixBlocks = 99
	s.finishDecision(BatchResult{RunningBatch: &Batch{}})
	if len(store.updates) != 1 || store.updates[0].MaxPrefixBlocks != 0 || len(p.feedback[0].RestoreUpdates) != 1 {
		t.Fatal("restore metadata not applied/recorded")
	}
	p.feedback[0].RestoreUpdates[0].MaxPrefixBlocks = 88
	if s.decision.record.Plan.Restores[0].MaxPrefixBlocks != 0 || s.decision.record.Feedback.RestoreUpdates[0].MaxPrefixBlocks != 0 {
		t.Fatal("policy mutated retained plan or feedback")
	}
}
