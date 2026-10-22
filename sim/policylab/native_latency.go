package policylab

import (
	"fmt"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

type nativeLatency struct{ store *kv.PeerCache }

func (n *nativeLatency) StepTime(batch []*sim.Request) int64 {
	if len(batch) == 0 {
		return 1
	}
	if len(batch) != 1 {
		panic("unmeasured batch size for native compute")
	}
	r := batch[0]
	prefix := r.ProgressIndex
	// On first admission BLIS records the cached prefix in ComputedTokens,
	// while ProgressIndex remains zero until the batch completes.
	if prefix == 0 {
		prefix = int64(len(n.store.GetCachedBlocks(r.FullInputTokens()))) * n.store.BlockSize()
	}
	ns, err := n.store.NativeForward(r.ID, prefix, int64(r.NumNewTokens))
	if err != nil {
		panic(fmt.Sprintf("request %s: %v", r.ID, err))
	}
	return max(1, (ns+999)/1000)
}

// The measured boundary has GPU-resident token IDs and ends at argmax.item().
// Tokenizer/HTTP/output streaming are outside this native experiment, not
// silently claimed to have zero cost in a serving deployment.
func (*nativeLatency) QueueingTime(*sim.Request) int64  { return 0 }
func (*nativeLatency) OutputTokenProcessingTime() int64 { return 0 }
func (*nativeLatency) PostDecodeFixedOverhead() int64   { return 0 }
