package sim

// KVExecutionDependency describes a physical prerequisite of an admitted
// request. Allocation success grants ownership, not permission to overwrite
// a source still being read by a STORE. AsyncBatchExecutor enforces these
// prerequisites even when execution observation is disabled.
type KVExecutionDependency struct {
	Kind        string  `json:"kind"`
	Transaction int64   `json:"transaction"`
	HBMBlocks   []int64 `json:"hbm_blocks"`
}

// KVExecutionDependencyStore returns an independent snapshot of outstanding
// physical dependencies. It exposes no predicted completion timestamps.
type KVExecutionDependencyStore interface {
	ExecutionDependencies(request string) []KVExecutionDependency
}
