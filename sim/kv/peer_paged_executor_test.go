package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

func pagedExecutorFixture(t *testing.T, configure ...func(*PagedExecutorConfig)) (*PagedExecutor, *sim.Request) {
	t.Helper()
	cost := kvruntime.TransferCost{SubmitNS: 2, DMANS: 3, HoldCPUUntilDMA: true}
	p := &kvruntime.Profile{QueueDepth: 8}
	for _, direction := range []string{"d2h", "h2d"} {
		p.Transfers = append(p.Transfers, kvruntime.TransferPoint{NUMA: 0, Layout: "paged", HostMemory: "pinned", Blocks: 1, Direction: direction, Cost: cost})
	}
	p.TorchCopies = &kvruntime.TorchCopyProfile{GPUSlots: 40, HostSlots: 64, Points: p.Transfers}
	p.InputPipeline = &kvruntime.InputPipelineProfile{GPUIndexBytes: 512, HostIndexBytes: 320, GPUAllocationQuantum: 512, ReplacedGPUInputBytes: 1536, Evidence: "test", NextInput: cost}
	for i := int64(1); i <= 40; i++ {
		p.InputPipeline.Index = append(p.InputPipeline.Index, kvruntime.InputIndexPoint{Count: i, PrepareNS: 5, Cost: cost})
	}
	weights := make([]float64, 30)
	for i := range weights {
		weights[i] = 1.0 / 30
	}
	p.Compute = []kvruntime.ComputePoint{{Length: 256, Mode: "resident", ForwardWeights: [][]float64{weights, weights}}}
	p.PagedForward = []kvruntime.PagedForwardPoint{{Prefix: 0, Query: 16, BeforeAllocatedBytes: 15 << 30, ReservedBytes: 16 << 30, Evidence: "test", Stages: map[string]kvruntime.BridgeStage{"forward": {WallNS: 30}, "observe_token": {WallNS: 2}, "scatter_delta": {WallNS: 57, SubmitNS: 1, ServiceNS: 1}, "hf_release": {WallNS: 1}}, PhasePeakDelta: map[string]int64{"forward": 1 << 20, "observe_token": 918272, "scatter_delta": 917760, "hf_release": 917760}, PhaseEndDelta: map[string]int64{"forward": 918272, "observe_token": 917760, "scatter_delta": 917760, "hf_release": 0}, Output: &kvruntime.OutputCopyPoint{Cost: cost, GPUStorageBytes: 512, HostStorageBytes: 8, HostReservedBytes: 8, ReferenceToken: 7, Evidence: "test"}}}
	req := &sim.Request{ID: "request", InputTokens: make([]sim.TokenID, 16), OutputTokens: []sim.TokenID{12345}}
	for i := range req.InputTokens {
		req.InputTokens[i] = sim.TokenID(i + 1)
	}
	config := PagedExecutorConfig{Profile: p, HBMSlots: 40, DRAMSlots: 64, CopyEnginePolicy: "fifo", Requests: []*sim.Request{req}}
	for _, apply := range configure {
		apply(&config)
	}
	x, err := NewPagedExecutor(config, nil)
	if err != nil {
		t.Fatal(err)
	}
	return x, req
}

func TestPagedExecutorServiceBindingsDelayRealPublication(t *testing.T) {
	x, req := pagedExecutorFixture(t, func(c *PagedExecutorConfig) {
		c.PagedServiceClasses.Forward = "forward_work"
		c.CopySubmitServiceClass = "copy_issue"
		c.CopyD2HServiceClass = "d2h_work"
	})
	e := x.Runtime.Engine
	if err := e.SetServiceRate("forward_work", 0); err != nil {
		t.Fatal(err)
	}
	batch, err := x.StartBatch(1, req.ID, 0, 16, []int64{0}, req.InputTokens)
	if err != nil {
		t.Fatal(err)
	}
	_ = e.At(1000, func() {
		if err := e.SetServiceRate("forward_work", 500000); err != nil {
			t.Fatal(err)
		}
	})
	if err = e.RunUntil(500); err != nil {
		t.Fatal(err)
	}
	if batch.Token != nil || batch.Step.CompleteNS != 0 {
		t.Fatal("bound forward class did not delay publication")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if batch.Token == nil || *batch.Token != 12345 || batch.Step.CompleteNS <= 1000 {
		t.Fatal("resumed worker lost output dependency")
	}
	h := hash.ComputeBlockHashes(16, req.InputTokens)[0]
	copy := ExternalTransfer{ID: 1, Source: "hbm/slot/0", Destination: "dram/slot/0", Hash: h, Bytes: 917504, Reason: "store"}
	if err := e.SetServiceRate("copy_issue", 500000); err != nil {
		t.Fatal(err)
	}
	if err := e.SetServiceRate("d2h_work", 0); err != nil {
		t.Fatal(err)
	}
	start := e.NowNS
	if _, err = x.StartTransfer(copy); err != nil {
		t.Fatal(err)
	}
	_ = e.At(start+1000, func() {
		if err := e.SetServiceRate("d2h_work", 500000); err != nil {
			t.Fatal(err)
		}
	})
	if err = e.RunUntil(start + 500); err != nil {
		t.Fatal(err)
	}
	if x.CheckTransfer(1, h, "dram/slot/0") == nil || x.CheckIdle("hbm/slot/0") == nil {
		t.Fatal("bound copy class published or released the paused transaction")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if err = x.CheckTransfer(1, h, "dram/slot/0"); err != nil {
		t.Fatal(err)
	}
	if err = x.ValidateDrained(); err != nil {
		t.Fatal(err)
	}
}

func TestPagedExecutorCopiesCausalPagesAndProtectsInflightStorage(t *testing.T) {
	x, req := pagedExecutorFixture(t)
	e := x.Runtime.Engine
	h := hash.ComputeBlockHashes(16, req.InputTokens)[0]
	e.MustAdd(kvruntime.Task{Name: "output_backlog", Resource: "copy_d2h", DurationNS: 10000})
	batch, err := x.StartBatch(1, req.ID, 0, 16, []int64{0}, req.InputTokens)
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(100); err != nil {
		t.Fatal(err)
	}
	if batch.Token != nil || batch.Step.CompleteNS != 0 || x.CheckContents(h, "hbm/slot/0") == nil {
		t.Fatal("output or reusable KV published before physical completion")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if batch.Token == nil || *batch.Token != 12345 || x.CheckContents(h, "hbm/slot/0") != nil {
		t.Fatal("paged computation did not produce independent reference contents")
	}
	copy := ExternalTransfer{ID: 1, Source: "hbm/slot/0", Destination: "dram/slot/0", Hash: h, Bytes: 917504, Reason: "store"}
	if _, err = x.StartTransfer(copy); err != nil {
		t.Fatal(err)
	}
	if err = x.InvalidateHBM(0); err == nil {
		t.Fatal("recycled a source pinned by the physical copy")
	}
	if err = x.CheckIdle("dram/slot/0"); err == nil {
		t.Fatal("recycled the active destination")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if err = x.CheckContents(h, "dram/slot/0"); err != nil {
		t.Fatal(err)
	}
	if err = x.InvalidateHBM(0); err != nil {
		t.Fatal(err)
	}
	if err = x.CheckContents(h, "hbm/slot/0"); err == nil {
		t.Fatal("recycled slot kept valid old KV")
	}
	copy.ID, copy.Source, copy.Destination, copy.Reason = 2, "dram/slot/0", "hbm/slot/7", "restore"
	if _, err = x.StartTransfer(copy); err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if err = x.CheckContents(h, "hbm/slot/7"); err != nil {
		t.Fatal(err)
	}
	copy.Reason = "promotion"
	if _, err = x.StartTransfer(copy); err == nil {
		t.Fatal("duplicate transaction accepted under another reason")
	}
	if err = x.ValidateDrained(); err != nil {
		t.Fatal(err)
	}
	x.Runtime.Memory.Buffers["hbm/slot/7"].Cells[0].Origin++
	copy.ID, copy.Source, copy.Destination = 3, "hbm/slot/7", "dram/slot/1"
	if _, err = x.StartTransfer(copy); err == nil {
		t.Fatal("corrupt causal source accepted")
	}
}

func TestPagedExecutorRejectsCommandTokenSubstitution(t *testing.T) {
	x, req := pagedExecutorFixture(t)
	tokens := append([]sim.TokenID(nil), req.InputTokens...)
	tokens[0]++
	if _, err := x.StartBatch(1, req.ID, 0, 16, []int64{0}, tokens); err == nil {
		t.Fatal("accepted command tokens that disagree with the independent reference")
	}
	if _, err := x.StartBatch(1, req.ID, 0, 16, []int64{0}, req.InputTokens); err != nil {
		t.Fatal("rejected command stranded worker", err)
	}
	if err := x.Runtime.Engine.Run(); err != nil {
		t.Fatal(err)
	}
}

func TestControlRoundWaitsForPagedWorkerOutputAndRelease(t *testing.T) {
	x, req := pagedExecutorFixture(t)
	e := x.Runtime.Engine
	p := &kvruntime.ControlProfile{Models: map[string]kvruntime.ControlCost{}}
	for _, name := range []string{"python_send", "request_available", "go_decode", "go_poll", "go_encode", "go_write", "reply_available", "python_decode", "batch_dispatch", "batch_queue_put", "batch_wake", "batch_entry", "batch_exit", "batch_notify"} {
		p.Models[name] = kvruntime.ControlCost{Coefficients: map[string]float64{"base": 10}}
	}
	e.MustAdd(kvruntime.Task{Name: "output_backlog", Resource: "copy_d2h", DurationNS: 10000})
	var body *PagedBatchResult
	var notified, released bool
	_, err := x.Runtime.ControlRound(kvruntime.ControlRoundSpec{ID: "physical-round", Profile: p, Decide: func() (*kvruntime.ControlDecision, error) {
		return &kvruntime.ControlDecision{Commands: []kvruntime.ControlCommand{{ID: "batch-1", Kind: "batch", Start: func() (int64, error) {
			var err error
			body, err = x.StartBatch(1, req.ID, 0, 16, []int64{0}, req.InputTokens)
			if err != nil {
				return 0, err
			}
			return body.Step.CompletionTask, nil
		}, Ready: func(at int64) {
			if body.Token == nil || *body.Token != 12345 || at <= body.Step.CompleteNS {
				t.Error("notification preceded actual output or worker exit")
			}
			if err := x.CheckIdle("hbm/slot/0"); err != nil {
				t.Error(err)
			}
			notified = true
		}}}}, nil
	}, Released: func(int64) { released = true }})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(1000); err != nil {
		t.Fatal(err)
	}
	if !released || notified || body == nil || body.Token != nil {
		t.Fatal("dispatch release confused with model completion")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if !notified {
		t.Fatal("missing completed worker notification")
	}
	if err = x.ValidateDrained(); err != nil {
		t.Fatal(err)
	}
}
