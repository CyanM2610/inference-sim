package policylab

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestConcurrentRestorePressureProgress(t *testing.T) {
	for seed := int64(0); seed < 12; seed++ {
		for _, policy := range []string{"lru_drop", "lfu_store"} {
			c := labConfig(1, "dram")
			c.Policy.Name = policy
			c.PrefillChunk = 32
			c.MaxSequences = 4
			c.Mechanisms = &kv.PeerMechanisms{BackgroundStorePool: "dram", GroupTransfers: true, ConcurrentRestores: true, RestoreWindow: 32}
			c.Requests = nil
			rng := rand.New(rand.NewSource(seed))
			for i := 0; i < 80; i++ {
				toks := make([]sim.TokenID, 17+rng.Intn(65))
				group := rng.Intn(5)
				for j := range toks {
					toks[j] = sim.TokenID(group*1000 + j)
				}
				c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i * 500), Input: toks, Output: make([]sim.TokenID, 1+rng.Intn(24))})
			}
			r, err := Run(c)
			if err != nil {
				t.Fatalf("seed=%d policy=%s: %v; HBM=%v", seed, policy, err, r.HBM)
			}
			loads := 0
			for _, event := range r.Events {
				if event.Name == "hbm_publish" && event.Reason == "restore" {
					loads++
				}
			}
			if loads == 0 {
				t.Fatal("pressure fixture did not exercise restores")
			}
			for _, pool := range r.Pools {
				if pool["read_pins"] != 0 || pool["reserved"] != 0 {
					t.Fatal("completed workload retained pending transfers")
				}
			}
		}
	}
}
