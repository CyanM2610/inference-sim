package policylab

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func labConfig(n int, topology string) Config {
	c := Config{Seed: 42, BlockTokens: 16, MaxBatchTokens: 128, MaxSequences: 2, PrefillChunk: 64, Model: sim.ModelConfig{NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 8, BytesPerParam: 2}, Hardware: sim.HardwareCalib{TFlopsPeak: 312, BwPeakTBs: 1.55, MfuPrefill: 0.5, MfuDecode: 0.3}, Alpha: []float64{0, 0, 0}, Routing: "round-robin", Policy: PolicyConfig{Name: "lfu_store", MinFrequency: 1}}
	for _, tier := range []string{"dram", "cxl"} {
		if topology != tier && topology != "both" {
			continue
		}
		c.Pools = append(c.Pools, kv.PeerPoolConfig{ID: tier, CapacityBlocks: 8})
		c.Resources = append(c.Resources, kv.PeerResource{ID: tier, BytesPerUS: 25000, LatencyUS: 3})
	}
	for i := 0; i < n; i++ {
		ic := InstanceConfig{HBMBlocks: 10}
		for _, p := range c.Pools {
			ic.Access = append(ic.Access, kv.PeerAccess{Pool: p.ID, ReadPath: []string{p.ID}, WritePath: []string{p.ID}})
		}
		c.Instances = append(c.Instances, ic)
	}
	for round := 0; round < 6; round++ {
		for i := 0; i < n; i++ {
			group := round % 3
			toks := make([]sim.TokenID, 65)
			for j := range toks {
				toks[j] = sim.TokenID(group*1000 + j + 1)
			}
			c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprintf("r%d_%d", round, i), At: int64(round) * 100000, Input: toks, Output: []sim.TokenID{9, 10}})
		}
	}
	return c
}
func TestPeerLabTopologyMatrix(t *testing.T) {
	for _, n := range []int{1, 2, 4} {
		for _, top := range []string{"hbm", "dram", "cxl", "both"} {
			t.Run(fmt.Sprintf("%d_%s", n, top), func(t *testing.T) {
				r, err := Run(labConfig(n, top))
				if err != nil {
					t.Fatal(err)
				}
				if len(r.Requests) != 6*n {
					t.Fatal("missing requests")
				}
				if top != "hbm" && r.Counts["l2_publish"] == 0 {
					t.Fatal("test did not exercise stores")
				}
			})
		}
	}
}
func TestPeerLabDeterminism(t *testing.T) {
	c := labConfig(2, "both")
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.Requests, b.Requests) {
		t.Fatal("nondeterministic output")
	}
}
func TestPeerLabPromotion(t *testing.T) {
	c := labConfig(1, "cxl")
	c.Promotions = []PromotionConfig{{Instance: 0, At: 290000, Tokens: c.Requests[0].Input, Budget: 1}}
	r, e := Run(c)
	if e != nil {
		t.Fatal(e)
	}
	n := 0
	for _, v := range r.Events {
		if v.Name == "hbm_publish" && v.Reason == "promotion" {
			n++
		}
	}
	if n == 0 {
		t.Fatal("promotion mode did no work")
	}
}
func TestPeerLabPressure(t *testing.T) {
	c := labConfig(4, "both")
	for i := range c.Requests {
		c.Requests[i].At = int64(i) * 100
	}
	if _, e := Run(c); e != nil {
		t.Fatal(e)
	}
}

func TestPeerLabExactCapacityFinalToken(t *testing.T) {
	c := labConfig(1, "hbm")
	c.Instances[0].HBMBlocks = 1
	c.Requests = c.Requests[:1]
	c.Requests[0].Input = c.Requests[0].Input[:16]
	c.Requests[0].Output = []sim.TokenID{9}
	if _, e := Run(c); e != nil {
		t.Fatal(e)
	} // final generated token needs no additional KV block
}

func TestPeerLabRandomPressureConservation(t *testing.T) {
	for seed := int64(0); seed < 12; seed++ {
		for _, policy := range []string{"lru_drop", "lfu_store", "ready_first", "cost_aware"} {
			c := labConfig(2, "both")
			c.Policy.Name = policy
			c.Seed = seed
			rng := rand.New(rand.NewSource(seed))
			c.Requests = nil
			for i := 0; i < 80; i++ {
				group := rng.Intn(5)
				length := 17 + rng.Intn(65)
				toks := make([]sim.TokenID, length)
				for j := range toks {
					toks[j] = sim.TokenID(group*1000 + j)
				}
				out := make([]sim.TokenID, 1+rng.Intn(24))
				for j := range out {
					out[j] = sim.TokenID(10000 + i*100 + j)
				}
				c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i * 500), Input: toks, Output: out})
			}
			r, e := Run(c)
			if e != nil {
				t.Fatalf("seed=%d policy=%s: %v; HBM: %v", seed, policy, e, r.HBM)
			}
			left := map[string]int64{}
			for _, v := range r.Events {
				if v.Name == "sim.RequestLeftEvent" {
					left[v.Request] = v.Time
				}
			}
			for _, q := range r.Requests {
				if math.Abs((float64(left[q.ID])-q.ArrivedAt*1e6)/1000-q.E2E) > 0.000001 {
					t.Fatalf("E2E does not match event completion for %s: %+v left=%d", q.ID, q, left[q.ID])
				}
			}
		}
	}
}
