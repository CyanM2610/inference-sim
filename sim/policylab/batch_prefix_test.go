package policylab

import "testing"

func TestBatchPrefixReuseChangesOnlyEligibleActualWork(t *testing.T) {
	for _, arrival := range []int64{0, 1} {
		c := producerLabConfig()
		c.Requests[1].At = arrival
		c.Requests[2].At = arrival
		c.BatchPrefixReuse = true
		r, err := Run(c)
		if err != nil {
			t.Fatal(err)
		}
		if r.BatchPrefixCostCoverage == "" {
			t.Fatal("new execution work has no cost coverage declaration")
		}
		for _, d := range r.PolicyDecisions {
			if !d.View.Capabilities.BatchPrefixReuseKnown || !d.View.Capabilities.BatchPrefixReuse {
				t.Fatal("enabled backend capability missing")
			}
		}
		var firstPrefix, queries int64
		seen := false
		for _, batch := range r.BatchShapes {
			for _, w := range batch.Scheduled {
				if w.Request == "b-consumer" {
					queries += w.Query
					if !seen {
						firstPrefix = w.Prefix
						seen = true
					}
				}
			}
		}
		want := int64(16)
		if arrival > 0 {
			want = 32
		}
		if !seen || firstPrefix != want || queries != 64-want {
			t.Fatal("wrong same-batch computation", arrival, firstPrefix, queries)
		}
		dependencies, published := 0, 0
		for _, e := range r.Events {
			if e.Name == "batch_prefix_dependency" {
				dependencies++
			}
			if e.Name == "batch_prefix_published" {
				published++
			}
		}
		if dependencies == 0 || dependencies != published || len(r.FinishedUS) != 3 {
			t.Fatal("missing dependency/publication/completion", dependencies, published)
		}
		for _, h := range r.HBM {
			if h["active_or_pinned"] != 0 {
				t.Fatal("HBM lease leaked")
			}
		}
		for _, p := range r.Pools {
			if p["reserved"] != 0 || p["read_pins"] != 0 {
				t.Fatal("pool lease leaked")
			}
		}
	}
}
