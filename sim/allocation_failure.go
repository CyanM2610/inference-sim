package sim

// AllocationFailure is an observed result of the last AllocateKVBlocks call,
// never a prediction of future free capacity. Wait includes reclaiming sources
// whose copies have not yet been adopted. Empty Kind means no known failure.
type AllocationFailure struct {
	// The observed offload lookup before this allocation attempt. A positive
	// candidate count does not promise allocation or transfer submission.
	RestoreLookupKnown     bool   `json:"restore_lookup_known,omitempty"`
	RestoreCandidateBlocks int64  `json:"restore_candidate_blocks,omitempty"`
	Request                string `json:"request"`
	Kind                   string `json:"kind"`
	Reason                 string `json:"reason"`
	NeededBlocks           int64  `json:"needed_blocks"`
	FreeBlocks             int64  `json:"free_blocks"`
	ReclaimingBlocks       int64  `json:"reclaiming_blocks"`
}

// AllocationFailureStore must overwrite its observation on every allocation
// attempt, including success. Callers consume it immediately after false.
type AllocationFailureStore interface {
	LastAllocationFailure() AllocationFailure
}
