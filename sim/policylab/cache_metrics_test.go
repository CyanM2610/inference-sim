package policylab

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestCacheMetricsMatchActualFirstCompute(t *testing.T) {
	for _, tier := range []string{"dram", "cxl"} {
		c := labConfig(1, tier)
		phaseTestConfig(&c)
		c.Mechanisms.BackgroundStorePool = tier
		c.TraceBatchShapes = true
		r, err := Run(c)
		if err != nil {
			t.Fatal(tier, err)
		}
		first := map[string]int64{}
		for _, b := range r.BatchShapes {
			for _, q := range b.Scheduled {
				if _, ok := first[q.Request]; !ok {
					first[q.Request] = q.Prefix
				}
			}
		}
		for _, row := range r.CacheMetrics.Requests {
			if !row.LookupObserved || !row.ReuseObserved || row.ReusedTokens != first[row.Request] {
				t.Fatalf("%s metric != actual first prefix: %+v vs %d", tier, row, first[row.Request])
			}
			if tier != "dram" && (row.DRAMHitTokens != 0 || row.DRAMReusedTokens != 0) {
				t.Fatal("other pool mislabeled as DRAM", row)
			}
		}
		if r.CacheMetrics.Totals.ReuseObservedRequests != int64(len(c.Requests)) {
			t.Fatal("missing request", r.CacheMetrics)
		}
	}
}

func TestCacheMetricsUnavailableAndEmptyDenominators(t *testing.T) {
	c := labConfig(1, "dram")
	m := collectCacheMetrics(c, nil)
	if m.Totals.Requests != 6 || m.Totals.LookupRequests != 0 || m.Totals.HBMDirectHitRate != nil {
		t.Fatal(m)
	}
	empty := kv.SummarizeCacheRequests(nil)
	if empty.PrefixReuseRate != nil || empty.DRAMHitRate != nil {
		t.Fatal(empty)
	}
}
