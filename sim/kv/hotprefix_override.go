package kv

import (
	"fmt"
	"reflect"
	"strings"

	simhash "github.com/inference-sim/inference-sim/sim/internal/hash"
)

// OneShotReclaim is an explicit offline intervention, not an online policy.
// The complete deterministic prefix is replayed; one matching legal group is
// selected and ordinary LRU decisions resume immediately afterwards.
type OneShotReclaim struct {
	Sequence         int64    `json:"sequence"`
	TimeUS           int64    `json:"time_us"`
	CandidatesSHA256 string   `json:"candidates_sha256"`
	MemberBlocks     []int64  `json:"member_blocks"`
	MemberHashes     []string `json:"member_hashes"`
}

func logicalCandidateDigest(groups []hotPrefixLogicalSegment, deficit int64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d\n", deficit)
	for _, g := range groups {
		for i, a := range g.members {
			fmt.Fprintf(&b, "%d:%s|", a.BlockID, g.hashes[i])
		}
		b.WriteByte('\n')
	}
	return simhash.DigestHex([]byte(b.String()))
}

func (p *HotPrefixPolicy) overrideLogicalChoice(groups []hotPrefixLogicalSegment, deficit int64, original int) int {
	p.overrideThisChoice = false
	x := p.config.OneShotReclaim
	if x == nil || p.diagnostics.Choices+1 != x.Sequence {
		return original
	}
	if logicalCandidateDigest(groups, deficit) != x.CandidatesSHA256 {
		panic("one-shot reclaim candidate snapshot diverged")
	}
	for i, g := range groups {
		var ids []int64
		for _, a := range g.members {
			ids = append(ids, a.BlockID)
		}
		if reflect.DeepEqual(ids, x.MemberBlocks) && reflect.DeepEqual(g.hashes, x.MemberHashes) {
			p.overrideThisChoice = true
			return i
		}
	}
	panic("one-shot reclaim target is not a legal candidate")
}
