package policylab

import (
	"reflect"
	"testing"
)

func TestBatchShapeTracePreservesExecution(t *testing.T) {
	c := labConfig(1, "dram")
	c.BatchCost = &BatchCostConfig{CoefficientsUS: []float64{100, 2, 0, 0}, Provenance: "analytic test"}
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.TraceBatchShapes = true
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) || !reflect.DeepEqual(a.FinishedUS, b.FinishedUS) {
		t.Fatal("observation changed execution")
	}
	if len(a.BatchShapes) != 0 || len(b.BatchShapes) == 0 {
		t.Fatal("trace opt-in is ineffective")
	}
	for _, row := range b.BatchShapes {
		queries := int64(0)
		for _, r := range row.Scheduled {
			if r.Prefix < 0 || r.Query <= 0 {
				t.Fatal("invalid shape")
			}
			queries += r.Query
		}
		if row.DurationUS != 100+2*queries {
			t.Fatal("trace differs from actual cost inputs")
		}
	}
}
