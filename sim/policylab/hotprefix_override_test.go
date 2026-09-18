package policylab

import (
	"crypto/sha256"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestOneShotPhysicalRunKeepsPrefixAndCompletes(t *testing.T) {
	c := futureLabConfig("lru")
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	var picked *kv.HotPrefixChoice
	for _, d := range a.HotPrefixDiagnostics["instance_0"].Records {
		if len(d.Candidates) > 1 {
			q := d
			picked = &q
			break
		}
	}
	if picked == nil {
		t.Fatal("fixture has no alternative")
	}
	var text strings.Builder
	fmt.Fprintf(&text, "%d\n", picked.DeficitBlocks)
	for _, g := range picked.Candidates {
		for i, id := range g.MemberBlocks {
			fmt.Fprintf(&text, "%d:%s|", id, g.MemberHashes[i])
		}
		text.WriteByte('\n')
	}
	var target kv.HotPrefixChoiceCandidate
	for _, g := range picked.Candidates {
		if g.BlockID != picked.SelectedBlock {
			target = g
			break
		}
	}
	c.HotPrefix.OneShotReclaim = &kv.OneShotReclaim{Sequence: picked.Sequence, TimeUS: picked.TimeUS, CandidatesSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(text.String()))), MemberBlocks: target.MemberBlocks, MemberHashes: target.MemberHashes}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, d := range b.HotPrefixDiagnostics["instance_0"].Records {
		if d.Sequence < picked.Sequence && !reflect.DeepEqual(d, a.HotPrefixDiagnostics["instance_0"].Records[d.Sequence-1]) {
			t.Fatal("decision prefix changed")
		}
		if d.OfflineOverride {
			count++
			if d.SelectedBlock != target.BlockID {
				t.Fatal("wrong physical group")
			}
		}
	}
	if count != 1 || len(b.FinishedUS) != len(c.Requests) || b.HBM["instance_0"]["total_refs"] != 0 {
		t.Fatal("incomplete or repeated intervention")
	}
}
