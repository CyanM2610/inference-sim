package sim

// NativeTransferObject is the immutable geometry of one host chunk. Staging
// always moves the full object; KernelBytes can exclude a restored HBM prefix.
// HostOffset is relative to the registered pool, not a physical host address.
type NativeTransferObject struct {
	Key          string              `json:"key"`
	Pool         string              `json:"pool"`
	HostOffset   int64               `json:"host_offset"`
	StagingBytes int64               `json:"staging_bytes"`
	KernelBytes  int64               `json:"kernel_bytes"`
	SkipTokens   int64               `json:"skip_tokens"`
	HBMBlocks    []int64             `json:"hbm_blocks"`
	Segments     []NativeCopySegment `json:"segments"`
}

type NativeCopySegment struct {
	HostOffset int64 `json:"host_offset"`
	Bytes      int64 `json:"bytes"`
}

type NativeTransferShape struct {
	Direction string                 `json:"direction"`
	Objects   []NativeTransferObject `json:"objects"`
}

// ModelComplete is a modeled-media boundary, not measured DMA completion.
// Publication must join both branches, followed by any explicit host service.
type NativeTransferPlan struct {
	Stages        []NativeStage `json:"stages"`
	ProducerWait  string        `json:"producer_wait"`
	GPUComplete   string        `json:"gpu_complete"`
	ModelComplete string        `json:"model_complete"`
	Published     string        `json:"published"`
	CostScope     string        `json:"cost_scope,omitempty"`
	CoverageGaps  []string      `json:"coverage_gaps,omitempty"`
	// Only failed submissions name this positive host-service milestone. A
	// failed receipt still joins GPUComplete, ModelComplete and this handler.
	FailureObserved string `json:"failure_observed,omitempty"`
	// Persistent per-context staging allocation, shared by STORE and LOAD.
	// This is not a new allocation or KV-pool debit for each operation; other
	// registration buffers and registration costs remain separate obligations.
	StagingSlots       int   `json:"registered_staging_slots,omitempty"`
	StagingBufferBytes int64 `json:"registered_staging_buffer_bytes,omitempty"`
}

type NativeTransferCost interface {
	TransferPlan(NativeTransferShape) NativeTransferPlan
}

// NativeTransferFailure separates acquired source/target geometry from the
// prefix of host objects actually submitted before an operation-level failure.
// No unsubmitted object may enter CUDA or modeled-media service. This is an
// execution-environment diagnostic, never a Placement-policy result.
type NativeTransferFailure struct {
	ReservedShape    NativeTransferShape `json:"reserved_shape"`
	SubmittedObjects int                 `json:"submitted_objects"`
}

type NativeTransferFailureCost interface {
	FailedTransferPlan(NativeTransferFailure) NativeTransferPlan
}
