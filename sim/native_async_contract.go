package sim

// NativeWork is an immutable submitted slice of a request, expressed in KV
// positions. KnownTokens is scheduler-visible history, excluding placeholders.
// The request pointer is runtime-only; cost/policy views contain shapes instead.
type NativeWork struct {
	Request                 *Request
	Generation              uint64
	Start, End, KnownTokens int64
	Sample                  bool
}

// NativeBatchKV owns captured physical IDs independently of the request's
// current allocation. BeforeConsume advances the global completed-step fence,
// drops capture references and drains retirement ownership before outputs are
// handled; Consumed finalizes publication/bookkeeping without freeing pages.
// Each boundary is exactly once, including empty connector-only iterations.
type NativeBatchKV interface {
	DeviceComplete(now int64)
	BeforeConsume(now int64)
	Consumed(now int64)
}

// NativePhasedBatchKV separates worker host submission/poll from GPU completion.
// PostForward may enqueue copies protected by an unresolved producer fence;
// Poll freezes the receipts returned by this worker iteration.
type NativePhasedBatchKV interface {
	NativeBatchKV
	KVAccessPlan() NativeKVAccessPlan
	Computed(now int64)
	PostForward(now int64, producer *NativeFence)
	Poll(now int64)
}

// NativeKVAccessDependency fences device access, not block-pool allocation.
// Fence comes from a submitted transfer's terminal publication. It cannot be
// replaced by a predicted absolute completion time supplied by a policy.
type NativeKVAccessDependency struct {
	Transaction  int64        `json:"transaction"`
	HBMBlocks    []int64      `json:"hbm_blocks"`
	PendingWrite bool         `json:"pending_write"`
	Fence        *NativeFence `json:"-"`
}

// Physical IDs are a runtime snapshot for causal audit, never a policy oracle.
type NativeKVAccessPlan struct {
	ReadBlocks   []int64                    `json:"read_blocks"`
	WriteBlocks  []int64                    `json:"write_blocks"`
	Dependencies []NativeKVAccessDependency `json:"dependencies"`
}

type NativePhasedKVStore interface {
	NativeAsyncKVStore
	EnableNativePhaseTransfers(*NativeExecutor, string) error
}

// NativeAsyncKVStore keeps allocation, content and physical lifetimes out of
// the scheduler. ReserveNativeRange never reads not-yet-emitted output tokens.
type NativeAsyncKVStore interface {
	KVStore
	EventDrivenKVStore
	EnableNativeAsyncKV() error
	// Mirrors the connector contract used by native preemption. Producer
	// hand-offs discard stale samples; best-effort caches can deliver them.
	NativeRequiresKVDelivery() bool
	ReserveNativeRange(req *Request, start, end int64, cached []int64) bool
	CaptureNativeBatch(work []NativeWork) NativeBatchKV
}

// NativeBatchShape is the online cost view. No actual output length or token
// contents are provided to a cost predictor.
type NativeBatchShape struct {
	Request           string `json:"request"`
	ContextTokens     int64  `json:"context_tokens"`
	QueryTokens       int64  `json:"query_tokens"`
	PromptQueryTokens int64  `json:"prompt_query_tokens"`
}

// NativeAsyncCost separates the host scheduler, serial worker compute envelope,
// output-ready delay and FIFO consumption. WorkerUS includes its launch gaps;
// callers must not add the same host submission interval a second time.
// These are cost interfaces, not a claim that a particular model is calibrated.
type NativeHostCost interface {
	ScheduleUS(waiting, running int) int64
	ConsumeUS(requests int) int64
}
type NativeAsyncCost interface {
	NativeHostCost
	WorkerUS([]NativeBatchShape) int64
	OutputUS([]NativeBatchShape) int64
}
