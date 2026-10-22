package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"testing"
)

func TestHashWorkCountsEncodedBytesWithoutChangingDigest(t *testing.T) {
	a, b := NewKVCacheState(2, 16), NewKVCacheState(2, 16)
	a.EnableHashWorkAccounting()
	tokens := []sim.TokenID{1, -2, 100}
	previous := "previous"
	want := hash.HashBlock(previous, tokens)
	if a.hashBlock(previous, tokens) != want || b.hashBlock(previous, tokens) != want {
		t.Fatal("accounting changed hash identity")
	}
	work := a.TakeHashWork()
	if work != (HashWork{Calls: 1, Tokens: 3, Bytes: 17, SHA256Blocks: 1}) {
		t.Fatalf("incorrect encoded work: %+v", work)
	}
	if a.TakeHashWork() != (HashWork{}) || b.TakeHashWork() != (HashWork{}) {
		t.Fatal("accounting was not per-cache and per-window")
	}
}
