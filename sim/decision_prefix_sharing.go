package sim

import "fmt"

// DecisionPrefixBlock identifies content, not a physical allocation. IDs are
// opaque and only comparable for equality within a snapshot. Ready/recoverable
// contiguous endpoints remain authoritative; identity alone promises no hit.
type DecisionPrefixBlock struct {
	ID          string `json:"id"`
	LoadPending bool   `json:"load_pending"`
}

type DecisionSharingStateProvider interface {
	DecisionStateProvider
	DecisionStateWithSharing([]DecisionKVQuery) DecisionKVState
}

func (s *Simulator) EnableDecisionPrefixSharing() error {
	if s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents {
		return fmt.Errorf("prefix sharing requires a pre-run decision policy and batch completion events")
	}
	p, ok := s.KVCache.(DecisionSharingStateProvider)
	if !ok || !p.DecisionStateWithSharing(nil).PrefixSharingKnown {
		return fmt.Errorf("prefix sharing requires an engine-phase sharing state provider")
	}
	s.decision.prefixSharing = true
	return nil
}

// decisionRestoreKeys excludes resident blocks. Existing in-flight loads block
// selection until adoption, including loads owned by a different request.
func decisionRestoreKeys(r DecisionKVRequest) ([]string, bool) {
	if r.LocalPrefixBlocks < 0 || r.RecoverablePrefixBlocks < r.LocalPrefixBlocks || r.RecoverablePrefixBlocks > int64(len(r.PrefixBlocks)) {
		panic("invalid shared prefix range")
	}
	var keys []string
	for _, block := range r.PrefixBlocks[r.LocalPrefixBlocks:r.RecoverablePrefixBlocks] {
		if block.ID == "" {
			panic("missing shared prefix identity")
		}
		if block.LoadPending {
			return nil, false
		}
		keys = append(keys, block.ID)
	}
	return keys, true
}
