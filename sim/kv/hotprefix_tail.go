package kv

// tailOptions enumerates a logarithmic set of block-aligned tail lengths.
// Every option comes from an already legal heat-coherent resident segment;
// overlapping alternatives are choices, never concurrent reclamation plans.
func tailOptions(groups []hotPrefixLogicalSegment, deficit int64) []hotPrefixLogicalSegment {
	if deficit <= 0 {
		panic("tail policy requires a positive current allocation deficit")
	}
	var options []hotPrefixLogicalSegment
	for _, group := range groups {
		limit := min(int64(len(group.members)), deficit)
		var resident []int64
		for _, a := range group.members {
			resident = append(resident, a.BlockID)
		}
		appendOption := func(length int64) {
			x := group
			x.members = group.members[:length]
			x.hashes = group.hashes[:length]
			x.residentMembers = resident
			options = append(options, x)
		}
		last := int64(0)
		for length := int64(1); length <= limit; {
			appendOption(length)
			last = length
			if length > limit/2 {
				break
			}
			length *= 2
		}
		if last != limit && limit > 0 {
			appendOption(limit)
		}
	}
	return options
}
