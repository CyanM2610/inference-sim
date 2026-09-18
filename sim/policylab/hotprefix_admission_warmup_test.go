package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestNativeAdmissionWarmupHasIdenticalHistoryBeforeTargetRules(t *testing.T) {
	t.Run("legacy_coupling", func(t *testing.T) { checkNativeAdmissionWarmup(t, nil) })
	t.Run("fixed_shadow", func(t *testing.T) { threshold := int64(2); checkNativeAdmissionWarmup(t, &threshold) })
}

func checkNativeAdmissionWarmup(t *testing.T, shadowThreshold *int64) {
	t.Helper()
	var prefix []kv.PeerRecord
	var boundary kv.PeerRecord
	for _, target := range []struct {
		rule      string
		threshold int64
	}{{"threshold", 1}, {"threshold", 2}, {"cost_next", 2}} {
		c := admissionLabConfig()
		// One slot becomes reserved before the reclaim batch finishes. Later
		// drops must share the same Shadow rules while that STORE is unready.
		c.Pools[0].CapacityBlocks = 1
		c.HotPrefix.AdmissionThreshold = target.threshold
		c.HotPrefix.ShadowThreshold = shadowThreshold
		c.HotPrefix.AdmissionCost = &kv.HotPrefixAdmissionConfig{Rule: target.rule,
			WarmupUntilFullReady: &kv.HotPrefixAdmissionWarmupConfig{AdmissionThreshold: 1}}
		out, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		index, switches := -1, 0
		for i, e := range out.Events {
			if e.AdmissionCost != nil && e.AdmissionCost.WarmupStateSHA256 != "" {
				index = i
				switches++
			}
		}
		if switches != 1 {
			t.Fatalf("fixture did not reach exactly one real warmup boundary: %d", switches)
		}
		e := out.Events[index]
		if e.AdmissionCost.WarmupReadyBlocks != c.Pools[0].CapacityBlocks || !e.AdmissionCost.PoolFull || e.AdmissionCost.Victim == "" {
			t.Fatal("boundary lacked a legal full READY pool")
		}
		if prefix == nil {
			prefix, boundary = out.Events[:index], e
		} else if !reflect.DeepEqual(prefix, out.Events[:index]) || e.Time != boundary.Time || e.Hash != boundary.Hash || e.Request != boundary.Request || e.AdmissionCost.WarmupStateSHA256 != boundary.AdmissionCost.WarmupStateSHA256 {
			t.Fatal("target rule contaminated shared executed warmup")
		}
		if target.rule == "threshold" && target.threshold == 1 {
			c.HotPrefix.AdmissionCost.WarmupUntilFullReady = nil
			baseline, err := Run(c)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(baseline.FirstTokenUS, out.FirstTokenUS) || !reflect.DeepEqual(baseline.FinishedUS, out.FinishedUS) || !reflect.DeepEqual(baseline.CacheMetrics, out.CacheMetrics) || !reflect.DeepEqual(baseline.Counts, out.Counts) {
				t.Fatal("threshold1 warmup observer changed threshold1 execution")
			}
		}
	}
}
