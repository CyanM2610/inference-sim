package kv

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

func TestPagedExecutorUsesIndependentInternalOutput(t *testing.T) {
	refs := map[string]map[string]uint64{"request": {"0/16": 777}}
	x, req := pagedExecutorFixture(t, func(c *PagedExecutorConfig) {
		for token := 17; token <= 32; token++ {
			c.Requests[0].InputTokens = append(c.Requests[0].InputTokens, sim.TokenID(token))
		}
		c.InternalOutputReferences = refs
	})
	// Caller mutation must not replace the request-dependent reference with the
	// unrelated shape-profile scalar (7) or another request's value.
	refs["request"]["0/16"] = 888
	batch, err := x.StartBatch(1, req.ID, 0, 16, []int64{0}, req.InputTokens[:16])
	if err != nil {
		t.Fatal(err)
	}
	if err = x.Runtime.Engine.Run(); err != nil {
		t.Fatal(err)
	}
	if value := x.Runtime.Memory.Buffers["native/host_output"].Cells[0].Origin; value != 777 {
		t.Fatalf("internal scalar = %d, want independent value 777", value)
	}
	if batch.Token != nil || batch.Step.OutputIndex != nil || batch.Step.RecomputedOutput {
		t.Fatal("internal prefill consumed the visible output budget")
	}
	if err = x.ValidateDrained(); err != nil {
		t.Fatal(err)
	}
}

func TestPagedExecutorRejectsMissingInternalOutputBeforePhysicalWork(t *testing.T) {
	x, req := pagedExecutorFixture(t, func(c *PagedExecutorConfig) {
		for token := 17; token <= 32; token++ {
			c.Requests[0].InputTokens = append(c.Requests[0].InputTokens, sim.TokenID(token))
		}
		c.InternalOutputReferences = map[string]map[string]uint64{}
	})
	var tasks int
	x.Runtime.Engine.Sink = func(r kvruntime.Record) {
		if r.Name == "task_created" {
			tasks++
		}
	}
	_, err := x.StartBatch(1, req.ID, 0, 16, []int64{0}, req.InputTokens[:16])
	if err == nil || !strings.Contains(err.Error(), "missing independent internal output") {
		t.Fatalf("missing reference was silently replaced: %v", err)
	}
	if tasks != 0 || x.native.forwardActive {
		t.Fatal("missing reference was rejected after starting physical work")
	}
	if err = x.ValidateDrained(); err != nil {
		t.Fatal(err)
	}
}

func TestPagedExecutorMarksRecomputedOutputAfterKVLoss(t *testing.T) {
	x, req := pagedExecutorFixture(t, func(c *PagedExecutorConfig) {
		c.Requests[0].OutputTokens = append(c.Requests[0].OutputTokens, 12346)
	})
	for id := int64(1); id <= 2; id++ {
		if id == 2 {
			if err := x.InvalidateHBM(0); err != nil {
				t.Fatal(err)
			}
		}
		batch, err := x.StartBatch(id, req.ID, 0, 16, []int64{0}, req.InputTokens)
		if err != nil {
			t.Fatal(err)
		}
		if batch.Token != nil || batch.Step.OutputIndex != nil {
			t.Fatal("output observation escaped before physical completion")
		}
		if err = x.Runtime.Engine.Run(); err != nil {
			t.Fatal(err)
		}
		if batch.Step.OutputIndex == nil || *batch.Step.OutputIndex != 0 || batch.Token == nil || *batch.Token != 12345 {
			t.Fatal("recompute lost independent output reference")
		}
		if batch.Step.RecomputedOutput != (id == 2) {
			t.Fatal("KV progress was confused with previously observed output progress")
		}
	}
	if err := x.ValidateDrained(); err != nil {
		t.Fatal(err)
	}
}
