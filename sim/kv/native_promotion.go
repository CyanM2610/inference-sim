package kv

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// NativePromotionPrefix is a copied, cache-bound snapshot of already visible
// text. Policies cannot mutate its content or provide hidden output identities.
type NativePromotionPrefix struct {
	owner  *PeerCache
	tokens []sim.TokenID
	hashes []string
}

type nativePromotion struct {
	id            string
	prefix        NativePromotionPrefix
	budget        int
	lookupReady   bool
	matched       int64
	ids           []int64
	cached        int
	settled       int
	failed        bool
	copiedTargets map[int64]bool
	claimed       bool
	submitted     bool
	cancelled     bool
	closing       bool
}

// SnapshotNativePromotionPrefix freezes a full-host-chunk prefix from current
// input and delivered output only. It includes the last full HBM block: unlike
// a frontend APC query, a control does not need to leave one token for logits.
func (s *PeerCache) SnapshotNativePromotionPrefix(req *sim.Request, tokens int64) (NativePromotionPrefix, error) {
	if s.fabric.hostChunks == nil || req == nil || req.Adapter != "" || req.IsMultimodal() ||
		tokens <= 0 || tokens%(s.fabric.hostChunks.blocks*s.BlockSizeTokens) != 0 || tokens > req.InputLen()+req.EmittedTokens() {
		return NativePromotionPrefix{}, fmt.Errorf("native promotion requires a visible whole-host-chunk text prefix")
	}
	input := append([]sim.TokenID(nil), req.InputTokens[:min(tokens, req.InputLen())]...)
	if tokens > req.InputLen() {
		count := tokens - req.InputLen()
		if count > int64(len(req.OutputTokens)) {
			return NativePromotionPrefix{}, fmt.Errorf("delivered output identity is unavailable")
		}
		input = append(input, req.OutputTokens[:count]...)
	}
	prefix := NativePromotionPrefix{owner: s, tokens: input}
	previous := ""
	for at := int64(0); at < tokens; at += s.BlockSizeTokens {
		previous = s.hashBlock(previous, input[at:at+s.BlockSizeTokens])
		prefix.hashes = append(prefix.hashes, previous)
	}
	return prefix, nil
}

// QueueNativePromotion starts an asynchronous source lookup, allocating no HBM.
// The single pending-control bound matches the native019 connector. Empty ID
// means a control is already pending; malformed/unsupported input is an error.
// A successful queue is not an allocation promise: frontend scheduling runs
// before whole-prefix capacity/budget admission in CaptureNativeBatch.
func (s *PeerCache) QueueNativePromotion(prefix NativePromotionPrefix, maxNewBlocks int) (string, error) {
	if !s.nativePhaseTransfers || !s.nativeExecutor.HasTransferModel() || !s.nativeExecutor.HasControlModel() ||
		prefix.owner != s || len(prefix.hashes) == 0 || maxNewBlocks <= 0 {
		return "", fmt.Errorf("native promotion requires a local snapshot, positive budget and explicit transfer/control costs")
	}
	if s.NativeControlPending() {
		return "", nil
	}
	s.nativeControlSerial++
	c := &nativePromotion{id: fmt.Sprintf("@native-promotion/%d", s.nativeControlSerial), prefix: prefix, budget: maxNewBlocks}
	s.nativeControl = c
	s.nativeControlRPCs++
	s.emit("native_control_queued", c.id, "", "host", "hbm", "lookup")
	s.nativeExecutor.SubmitControlRPC(s.clock, c.id, s.controlShape(c, "lookup"),
		func(at int64) {
			s.SetClock(at)
			lookup := s.ensureHostLookup(c.id, c.prefix.tokens)
			if lookup != nil {
				c.matched = int64(len(lookup.prefix)) * s.fabric.hostChunks.blocks * s.BlockSizeTokens
			} // A guarded END may have overtaken this LOOKUP: acquire no lease.
		}, func(at int64) {
			s.SetClock(at)
			s.nativeControlRPCs--
			c.lookupReady = true
			s.emit("native_control_lookup_ready", c.id, "", "host", "hbm", "result_visible")
			s.wake(at)
		})
	s.wake(s.clock)
	return c.id, nil
}

// CancelNativePromotion cancels selection, not captured work. Once captured,
// even a not-yet-submitted control keeps all HBM/source ownership until a real
// terminal receipt is polled and consumed. Unknown/retired IDs are no-ops.
func (s *PeerCache) CancelNativePromotion(id string) {
	c := s.nativeControl
	if c == nil || c.id != id || c.cancelled {
		return
	}
	c.cancelled = true
	s.emit("native_control_cancelled", id, "", "", "", "selection_cancelled")
	if len(c.ids) == 0 {
		s.retireNativePromotion(c, "cancel_before_capture")
	}
}

func (s *PeerCache) controlShape(c *nativePromotion, operation string) sim.NativeControlRPCShape {
	return sim.NativeControlRPCShape{Operation: operation, Tokens: int64(len(c.prefix.tokens)), Objects: int64(len(c.prefix.hashes)) / s.fabric.hostChunks.blocks}
}

// Runs only after the scheduler has reserved all foreground work. A rejection
// changes no HBM references, content, or free-list order, including cached free
// prefix pages which must not also be counted as missing-suffix capacity.
func (s *PeerCache) prepareNativePromotion() *nativePromotion {
	c := s.nativeControl
	if c == nil || !c.lookupReady || len(c.ids) != 0 || c.closing {
		return nil
	}
	if c.matched < int64(len(c.prefix.tokens)) {
		s.retireNativePromotion(c, "host_miss")
		return nil
	}
	var cached []int64
	freeCached := int64(0)
	for _, hash := range c.prefix.hashes {
		id, ok := s.lookup[hash]
		if !ok {
			break
		}
		cached = append(cached, id)
		if s.Blocks[id].RefCount == 0 {
			freeCached++
		}
	}
	missing := len(c.prefix.hashes) - len(cached)
	if missing == 0 || missing > c.budget || int64(missing) > s.FreeBlockCnt-freeCached {
		s.retireNativePromotion(c, "hbm_hit_budget_or_capacity")
		return nil
	}
	if s.nativePartialFailure > 0 {
		objects := (len(c.prefix.hashes)-1)/int(s.fabric.hostChunks.blocks) - len(cached)/int(s.fabric.hostChunks.blocks) + 1
		if s.nativePartialFailure >= objects {
			s.retireNativePromotion(c, "partial_failure_requires_multiple_retained_objects")
			return nil
		}
	}
	c.cached = len(cached)
	c.ids = append(c.ids, cached...)
	for _, id := range cached {
		s.pin(s.Blocks[id])
	}
	for i := 0; i < missing; i++ {
		b := s.popFreeBlock()
		if b == nil {
			panic("prechecked native control allocation failed")
		}
		if b.Hash != "" {
			s.emit("hbm_drop", c.id, b.Hash, "hbm", "", "native_control_lru")
		}
		s.invalidate(b)
		b.RefCount, b.InUse = 1, true
		s.pendingTargets[b.ID] = true
		c.ids = append(c.ids, b.ID)
	}
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "native_control_captured", Instance: s.id, Request: c.id,
		HBMBlocks: append([]int64(nil), c.ids...), Hashes: append([]string(nil), c.prefix.hashes...),
		Counters: map[string]int64{"cached_blocks": int64(c.cached), "new_blocks": int64(missing)}})
	return c
}

func (s *PeerCache) submitNativePromotion(c *nativePromotion) {
	if c != s.nativeControl || c.submitted || len(c.ids) == 0 || c.closing {
		panic("invalid native control submission")
	}
	c.submitted = true
	s.fabric.beginTransferGroup()
	for i := c.cached; i < len(c.ids); i++ {
		hash := c.prefix.hashes[i]
		entry, access := s.findForRestore(c.id, hash, c.prefix.tokens)
		if entry == nil || !s.fetchInto(hash, c.id, entry, access, nil, true, false, s.Blocks[c.ids[i]], c) {
			panic("native control lost its leased source or independent target")
		}
	}
	s.fabric.endTransferGroup(s.clock)
}

func (s *PeerCache) adoptNativePromotion(c *nativePromotion) {
	if c != s.nativeControl || !c.claimed || c.settled != len(c.ids)-c.cached || c.closing {
		panic("native control adopted without terminal ownership")
	}
	for i := c.cached; i < len(c.ids); i++ {
		b := s.Blocks[c.ids[i]]
		if b.RefCount != 1 || b.Hash != "" {
			panic("native control target was exposed before adoption")
		}
		hash := c.prefix.hashes[i]
		delete(s.restoring, s.restoreIdentity(c.id, hash))
		if !c.cancelled && !c.failed {
			b.Hash = hash
			s.HashToBlock[hash] = b.ID
			s.publishReadyCopy(b)
			s.promoted[b.ID] = true
			s.emit("hbm_publish", c.id, hash, "host", "hbm", "native_promotion_adopted")
		} else {
			s.invalidate(b)
		}
	}
	// One reverse-order release call preserves the native pool's partitioning
	// and LRU order for the entire reservation, including the old cached prefix.
	s.releaseNativeReferences(c.ids)
	reason := "success"
	if c.cancelled {
		reason = "cancelled_after_capture"
	}
	if c.failed {
		reason = "partial_submission_failed"
	}
	s.emit("native_control_adopted", c.id, "", "host", "hbm", reason)
	s.retireNativePromotion(c, reason)
}

func (s *PeerCache) retireNativePromotion(c *nativePromotion, reason string) {
	if c.closing || s.nativeControl != c {
		panic("native control retired twice")
	}
	c.closing = true
	s.nativeControl = nil
	// Installed native hybrid end_session hands cleanup to a background owner;
	// it does not keep has_pending_push_work true while waiting for END ACK.
	s.nativeControlRPCs++
	// ConfirmedSessionCleanup sends guarded END without waiting for LOOKUP.
	// The server's ended-request tombstone, not a client-side dependency, keeps
	// a later LOOKUP from reacquiring references after cancellation.
	s.nativeExecutor.SubmitControlRPC(s.clock, c.id, s.controlShape(c, "end"),
		func(at int64) { s.SetClock(at); s.endHostLookup(c.id) },
		func(at int64) {
			s.SetClock(at)
			s.nativeControlRPCs--
			s.emit("native_control_retired", c.id, "", "host", "", reason)
		})
}
