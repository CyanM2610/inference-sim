package kv

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// A single-server hybrid L1 lookup acquires source leases before acknowledging
// LOOKUP. STATUS is a separate round trip, submitted and consumed only when the
// scheduler calls the connector. Even a zero-chunk miss follows this protocol.
type nativeRequestLookup struct {
	id                    string
	input                 []sim.TokenID
	acked, query, replied bool
	accepted              bool
	visible, closing      bool
	matched               int64
	result                int64
	pending               int
}

// EnableNativeRequestLookups selects the frozen native connector's non-eager,
// nonblocking single-server path. Reduced synchronous diagnostics remain
// available by leaving it disabled, and must not claim native equivalence.
func (s *PeerCache) EnableNativeRequestLookups() error {
	if !s.nativePhaseTransfers || s.nativeExecutor == nil || !s.nativeExecutor.HasRequestRPCModel() ||
		s.nativeRequestLookups != nil || s.nativeControlSerial != 0 || s.nativeScheduledStep != 0 || len(s.RequestMap) != 0 || len(s.fabric.hostChunks.lookups) != 0 {
		return fmt.Errorf("native request lookups require unused phased runtime and explicit request RPC costs")
	}
	s.nativeRequestLookups = map[string]*nativeRequestLookup{}
	return nil
}

func (s *PeerCache) requestRPC(c *nativeRequestLookup, operation string, applied, replied func(int64)) {
	s.nativeRequestRPCSerial++
	id := fmt.Sprintf("%d", s.nativeRequestRPCSerial)
	shape := sim.NativeControlRPCShape{Operation: operation}
	if operation == "lookup" {
		shape.Tokens = int64(len(c.input))
		shape.Objects = shape.Tokens / (s.fabric.hostChunks.blocks * s.BlockSizeTokens)
	}
	s.nativeRequestRPCs++
	c.pending++
	s.nativeExecutor.SubmitRequestRPC(s.clock, id, c.id, shape, applied, func(at int64) {
		s.SetClock(at)
		s.nativeRequestRPCs--
		c.pending--
		replied(at)
		if c.closing && c.pending == 0 {
			delete(s.nativeRequestLookups, c.id)
			s.emit("native_request_lookup_retired", c.id, "", "host", "", "all_rpc_replies_drained")
		}
	})
}

func (s *PeerCache) NativeRequestLookupReady(req *sim.Request, preempted bool) bool {
	if s.nativeRequestLookups == nil || preempted {
		return true
	}
	c := s.nativeRequestLookups[req.ID]
	if c == nil {
		if s.fabric.hostChunks.ended[req.ID] {
			panic("ended native request re-entered waiting lookup")
		}
		c = &nativeRequestLookup{id: req.ID, input: append([]sim.TokenID(nil), req.PrefillTokens()...)}
		s.nativeRequestLookups[req.ID] = c
		s.emit("native_request_lookup_submitted", c.id, "", "", "host", "scheduler_callback")
		s.requestRPC(c, "lookup", func(at int64) {
			s.SetClock(at)
			if lookup := s.ensureHostLookup(c.id, c.input); lookup != nil {
				c.accepted = true
				c.matched = int64(len(lookup.prefix)) * s.fabric.hostChunks.blocks * s.BlockSizeTokens
			}
			s.emit("native_request_lookup_applied", c.id, "", "host", "", "directory_snapshot")
		}, func(at int64) {
			c.acked = true
			accepted := int64(0)
			if c.accepted {
				accepted = 1
			}
			s.fabric.emit(PeerRecord{Time: at, Name: "native_request_lookup_ack", Instance: s.id, Request: c.id,
				Counters: map[string]int64{"accepted": accepted}})
		})
	}
	if c.closing {
		panic("closing native request re-entered waiting lookup")
	}
	if c.visible {
		return true
	}
	if !c.acked {
		return false
	}
	if !c.accepted {
		panic("live request received a rejected lookup without modeled failure handling")
	}
	if !c.query {
		c.query = true
		s.emit("native_request_status_submitted", c.id, "", "", "host", "scheduler_callback_after_ack")
		s.requestRPC(c, "status", func(at int64) {
			s.SetClock(at)
			c.result = c.matched
			if s.fabric.hostChunks.ended[c.id] {
				c.result = 0 // guarded END may overtake this status request
			}
			s.fabric.emit(PeerRecord{Time: at, Name: "native_request_status_applied", Instance: s.id, Request: c.id,
				Counters: map[string]int64{"matched_tokens": c.result}})
		}, func(at int64) {
			c.replied = true
			s.emit("native_request_status_reply", c.id, "", "host", "", "future_available")
		})
	}
	if !c.replied {
		return false
	}
	c.visible = true
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "native_request_lookup_adopted", Instance: s.id, Request: c.id,
		Counters: map[string]int64{"matched_tokens": c.result}})
	return true
}

// Scheduler cancellation and completion submit END without waiting for LOOKUP
// or STATUS. The host tombstone prevents a late LOOKUP from reacquiring refs.
// In-flight LOAD physical reservations remain owned by the transfer itself.
func (s *PeerCache) endRequestHostLookup(id string) {
	c := s.nativeRequestLookups[id]
	if c == nil {
		s.endHostLookup(id)
		return
	}
	if c.closing {
		return
	}
	c.closing = true
	s.emit("native_request_end_submitted", c.id, "", "", "host", "request_finished")
	s.requestRPC(c, "end", func(at int64) {
		s.SetClock(at)
		s.endHostLookup(c.id)
		s.emit("native_request_end_applied", c.id, "", "host", "", "session_tombstone")
	}, func(at int64) {
		s.emit("native_request_end_reply", c.id, "", "host", "", "end_acknowledged")
	})
}
