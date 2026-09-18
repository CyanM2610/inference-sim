package policylab

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestHotPrefixDiagnosticsLeaveNativeExecutionUnchanged(t *testing.T) {
	c := nativeLabConfig()
	c.HotPrefix.PromotionBlocks = 0
	c.Requests = nil
	for i := 0; i < 12; i++ {
		input := make([]sim.TokenID, 65)
		for j := range input {
			input[j] = sim.TokenID((i%4)*1000 + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i) * 100000, Input: input, Output: []sim.TokenID{9, 8}})
	}
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.HotPrefix.Diagnostics = &kv.HotPrefixDiagnosticsConfig{TraceCandidates: true, MeasureCPU: true}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.BatchShapes, b.BatchShapes) ||
		!reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) || !reflect.DeepEqual(a.FinishedUS, b.FinishedUS) || !reflect.DeepEqual(a.CacheMetrics, b.CacheMetrics) {
		t.Fatal("instrumentation changed execution")
	}
	d := b.HotPrefixDiagnostics["instance_0"]
	if d == nil || d.Choices == 0 || len(d.Records) != int(d.Choices) {
		t.Fatal("fixture did not exercise diagnostics")
	}
	for i, r := range d.Records {
		if r.Sequence != int64(i+1) || r.Request == "" || r.DeficitBlocks <= 0 {
			t.Fatal("missing runtime context", r)
		}
	}
}
