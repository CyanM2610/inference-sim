package kv

import "slices"

// Batching changes submission granularity, not physical storage ownership.
// Every member retains its own reservation and completion callback; a group
// pays one job setup and transfers the sum of member bytes on its path.
func (f *PeerFabric) beginTransferGroup() {
	if f.native != nil || f.external != nil {
		panic("grouped abstract transfer on detailed runtime")
	}
	f.groupDepth++
}

func (f *PeerFabric) endTransferGroup(now int64) {
	f.groupDepth--
	if f.groupDepth < 0 {
		panic("unbalanced transfer group")
	}
	if f.groupDepth > 0 {
		return
	}
	buffer := f.groupBuffer
	f.groupBuffer = nil
	f.joinHostLoads(buffer)
	// A native MP operation spans all media in one request's object sequence.
	// Its execution model knows the per-object pool; path-only shims still need
	// homogeneous physical paths and are not the calibrated native interface.
	nativeOperation := len(f.stores) == 1 && f.stores[0].nativePhaseTransfers && f.stores[0].nativeExecutor.HasTransferModel()
	for len(buffer) > 0 {
		first := buffer[0]
		members := []*peerJob{first}
		remaining := buffer[:0]
		for _, j := range buffer[1:] {
			samePath := first.record.Source == j.record.Source && first.record.Destination == j.record.Destination && slices.Equal(first.path, j.path)
			sameNativeDirection := nativeOperation && (first.record.Destination == "hbm") == (j.record.Destination == "hbm")
			if first.prepare == nil && j.prepare == nil && first.cancel == nil && j.cancel == nil &&
				first.record.Request == j.record.Request && first.record.Instance == j.record.Instance &&
				first.record.Reason == j.record.Reason && (samePath || sameNativeDirection) {
				members = append(members, j)
			} else {
				remaining = append(remaining, j)
			}
		}
		buffer = remaining
		if len(members) == 1 {
			f.accountHostTransfer(first, members)
			f.submit(now, first)
			continue
		}
		record := first.record
		record.Hash = ""
		for _, j := range members {
			record.Hashes = append(record.Hashes, j.record.Hash)
			if j.record.Source != first.record.Source {
				record.Source = "l1"
			}
			if j.record.Destination != first.record.Destination {
				record.Destination = "l1"
			}
		}
		// The grouped copy must retain every member's physical source lease.
		record.HBMBlocks = nil
		for _, j := range members {
			record.HBMBlocks = append(record.HBMBlocks, j.record.HBMBlocks...)
		}
		group := &peerJob{nativeControl: first.nativeControl, record: record, units: int64(len(members)), path: first.path, schedule: first.schedule,
			done: func(at int64) {
				for _, j := range members {
					j.done(at)
				}
			},
			blockedRequests: func() []string {
				seen := map[string]bool{}
				for _, j := range members {
					if j.blockedRequests != nil {
						for _, id := range j.blockedRequests() {
							seen[id] = true
						}
					}
				}
				return sortedPeerIDs(seen)
			},
			deadline: func() int64 {
				d := first.effectiveDeadline(f)
				for _, j := range members[1:] {
					d = min(d, j.effectiveDeadline(f))
				}
				return d
			},
		}
		if record.Source == "l1" || record.Destination == "l1" {
			group.path = nil // NativeObjects carry every real pool/path identity.
		}
		for _, j := range members {
			group.overwrites = append(group.overwrites, j.overwrites...)
		}
		f.accountHostTransfer(group, members)
		f.submit(now, group)
	}
}
