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
// current allocation. DeviceComplete may submit STORE; Consumed releases the
// batch's pins after FIFO host consumption. Both are exactly-once operations.
type NativeBatchKV interface {
	DeviceComplete(now int64)
	Consumed(now int64)
}

// NativeAsyncKVStore keeps allocation, content and physical lifetimes out of
// the scheduler. ReserveNativeRange never reads not-yet-emitted output tokens.
type NativeAsyncKVStore interface {
	KVStore
	EventDrivenKVStore
	EnableNativeAsyncKV() error
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
type NativeAsyncCost interface {
	ScheduleUS(waiting, running int) int64
	WorkerUS([]NativeBatchShape) int64
	OutputUS([]NativeBatchShape) int64
	ConsumeUS(requests int) int64
}
