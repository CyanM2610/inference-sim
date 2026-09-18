package kv

import "fmt"

// EnableFirstPublishedCopy preserves the pinned native APC's insertion order
// among identical completed blocks. It does not merge physical blocks or refs.
// The legacy single-entry index remains the default for frozen experiments.
func (s *PeerCache) EnableFirstPublishedCopy() error {
	if s.prefixCopies != nil || len(s.ready) != 0 || len(s.RequestMap) != 0 || s.fabric.Pending() != 0 {
		return fmt.Errorf("prefix copy selection must be configured before cache use")
	}
	s.prefixCopies = map[string][]int64{}
	return nil
}

// publishReadyCopy is called only when compute/restore completion is visible.
// Re-observing a completed block must not move it behind a newer duplicate.
func (s *PeerCache) publishReadyCopy(b *KVBlock) {
	s.metricPublish(b)
	if s.hotprefix != nil {
		s.expireHotPrefixShadows()
		delete(s.hotprefix.shadows, b.Hash)
		s.hotprefix.policy.published(b.Hash)
	}
	if s.prefixCopies == nil {
		s.ready[b.ID] = b.Hash
		s.lookup[b.Hash] = b.ID
		return
	}
	if old, exists := s.ready[b.ID]; exists {
		if old != b.Hash {
			panic("completed prefix overwritten without invalidation")
		}
	} else {
		s.prefixCopies[b.Hash] = append(s.prefixCopies[b.Hash], b.ID)
	}
	s.ready[b.ID] = b.Hash
	s.lookup[b.Hash] = s.prefixCopies[b.Hash][0]
	// This embedded index is not the readiness authority; the underlying
	// allocator may hash allocated tokens before publication. Peer queries use
	// lookup, while this projection makes completed-state diagnostics consistent.
	s.HashToBlock[b.Hash] = s.lookup[b.Hash]
}

func (s *PeerCache) forgetReadyCopy(b *KVBlock) {
	if s.prefixCopies == nil {
		if id, ok := s.lookup[b.Hash]; ok && id == b.ID {
			delete(s.lookup, b.Hash)
		}
		delete(s.ready, b.ID)
		return
	}
	hash, exists := s.ready[b.ID]
	if !exists {
		return // an allocated but never published block
	}
	copies := s.prefixCopies[hash]
	index := -1
	for i, id := range copies {
		if id == b.ID {
			index = i
			break
		}
	}
	if index < 0 {
		panic("completed prefix missing from duplicate index")
	}
	copies = append(copies[:index], copies[index+1:]...)
	delete(s.ready, b.ID)
	if len(copies) == 0 {
		delete(s.prefixCopies, hash)
		delete(s.lookup, hash)
		delete(s.HashToBlock, hash)
	} else {
		s.prefixCopies[hash] = copies
		s.lookup[hash] = copies[0]
		s.HashToBlock[hash] = copies[0]
	}
}
