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
	for len(buffer) > 0 {
		first := buffer[0]
		members := []*peerJob{first}
		remaining := buffer[:0]
		for _, j := range buffer[1:] {
			if first.prepare == nil && j.prepare == nil && first.cancel == nil && j.cancel == nil &&
				first.record.Request == j.record.Request && first.record.Instance == j.record.Instance &&
				first.record.Reason == j.record.Reason && first.record.Source == j.record.Source &&
				first.record.Destination == j.record.Destination && slices.Equal(first.path, j.path) {
				members = append(members, j)
			} else {
				remaining = append(remaining, j)
			}
		}
		buffer = remaining
		if len(members) == 1 {
			f.submit(now, first)
			continue
		}
		record := first.record
		record.Hash = ""
		for _, j := range members {
			record.Hashes = append(record.Hashes, j.record.Hash)
		}
		// The grouped copy must retain every member's physical source lease.
		record.HBMBlocks = nil
		for _, j := range members {
			record.HBMBlocks = append(record.HBMBlocks, j.record.HBMBlocks...)
		}
		group := &peerJob{record: record, units: int64(len(members)), path: first.path, schedule: first.schedule,
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
		for _, j := range members {
			group.overwrites = append(group.overwrites, j.overwrites...)
		}
		f.submit(now, group)
	}
}
