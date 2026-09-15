package kv

import (
	"strconv"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// HashWork counts actual policy hash invocations and their encoded inputs.
// Accounting is optional, per-cache, and must not be included in native timing
// calibration. It changes neither hashing, caching nor scheduling decisions.
type HashWork struct {
	Calls        int64 `json:"calls"`
	Tokens       int64 `json:"tokens"`
	Bytes        int64 `json:"bytes"`
	SHA256Blocks int64 `json:"sha256_blocks"`
}

func (k *KVCacheState) EnableHashWorkAccounting() { k.hashAccounting = true }
func (k *KVCacheState) TakeHashWork() HashWork {
	work := k.hashWork
	k.hashWork = HashWork{}
	return work
}
func (k *KVCacheState) hashBlock(previous string, tokens []sim.TokenID) string {
	if k.hashAccounting {
		bytes := int64(len(previous))
		var buf [20]byte
		for _, token := range tokens {
			bytes += int64(len(strconv.AppendInt(buf[:0], int64(token), 10))) + 1
		}
		k.hashWork.Calls++
		k.hashWork.Tokens += int64(len(tokens))
		k.hashWork.Bytes += bytes
		k.hashWork.SHA256Blocks += (bytes + 9 + 63) / 64 // SHA-256 terminator and 64-bit length
	}
	return hash.HashBlock(previous, tokens)
}
