package sim

import "testing"

func copyProfileFixture() (NativeCopyProfile, NativeTransferExecution, NativeTransferObject) {
	p := NativeCopyProfile{Schema: 1, ObjectBytes: 1024, ChunkTokens: 16, BlockTokens: 4, SegmentBytes: 4096, StoreBatch: 1, LoadBatch: 2, Curves: map[string]NativeCopyCurve{}, Modeled: NativeModeledMedia{ReadGBPS: 1, WriteGBPS: 1, ReadNS: 5000, WriteNS: 5000}}
	for _, direction := range []string{"store", "load"} {
		for _, prefix := range []string{"kernel/gpu/", "staging/dram/", "staging/cxl_sim/"} {
			p.Curves[prefix+direction] = NativeCopyCurve{Coefficients: []float64{0, 1 << 20, 0}, MinBytes: 768, MaxBytes: 2048, MinObjects: 1, MaxObjects: 2, MaxExtra: 1}
		}
	}
	x := NativeTransferExecution{PoolMedia: map[string]string{"cold": "cxl_sim"}, HostResource: "cpu", ComputeResource: "compute", CopyResource: "copy", ReadModelResource: "read", WriteModelResource: "write", HostSubmitUS: 3, ModelRPCUS: 2, ReplyUS: 1, Scope: "artificial CPU test"}
	o := NativeTransferObject{Key: "object", Pool: "cold", HostOffset: 3584, StagingBytes: 1024, KernelBytes: 768, SkipTokens: 4, HBMBlocks: []int64{1, 2, 3}, Segments: []NativeCopySegment{{3584, 512}, {4096, 512}}}
	return p, x, o
}

func TestNativeCopyProfileChargesWholeStagingAndPartialScatter(t *testing.T) {
	p, x, o := copyProfileFixture()
	plan := p.TransferPlan(NativeTransferShape{Direction: "load", Objects: []NativeTransferObject{o}}, x)
	stages := map[string]NativeStage{}
	for _, s := range plan.Stages {
		stages[s.Name] = s
	}
	if stages["staging_0"].DurationUS != 1024 || stages["kernel_0"].DurationUS != 768 || plan.ProducerWait != "staging_0" || plan.GPUComplete != "kernel_0" {
		t.Fatal("partial LOAD undercharged staging or changed native order", plan)
	}
	if stages["model_serial_0"].DurationUS != 2 || stages["model_tail_0"].DurationUS != 5 || !stages["model_tail_0"].Delay || stages["model_tail_0"].Resource != "" {
		t.Fatal("modeled latency incorrectly occupies the modeled bandwidth resource", plan)
	}
	if len(plan.CoverageGaps) != 0 {
		t.Fatal(plan.CoverageGaps)
	}
	// Changing skip beyond measured geometry remains executable, but annotated.
	o.SkipTokens, o.KernelBytes, o.HBMBlocks = 12, 256, []int64{3}
	plan = p.TransferPlan(NativeTransferShape{Direction: "load", Objects: []NativeTransferObject{o}}, x)
	if len(plan.CoverageGaps) != 1 {
		t.Fatal("extrapolation was silently treated as measured", plan)
	}
}

func TestNativeCopyProfileStoreUsesOneBufferSlotPerChunk(t *testing.T) {
	p, x, o := copyProfileFixture()
	o.SkipTokens, o.KernelBytes, o.HBMBlocks = 0, 1024, []int64{1, 2, 3, 4}
	o2 := o
	o2.Key = "other"
	plan := p.TransferPlan(NativeTransferShape{Direction: "store", Objects: []NativeTransferObject{o, o2}}, x)
	if plan.ProducerWait != "kernel_0" || plan.GPUComplete != "staging_1" {
		t.Fatal(plan)
	}
	var cuda []string
	for _, s := range plan.Stages {
		if s.Stream == "native-kv-stream" {
			cuda = append(cuda, s.Name)
		}
	}
	if len(cuda) != 4 || cuda[0] != "kernel_0" || cuda[1] != "staging_0" || cuda[2] != "kernel_1" || cuda[3] != "staging_1" {
		t.Fatal("STORE was incorrectly merged into the LOAD batch size", cuda)
	}
}

func TestNativeCopyProfileDoesNotTurnTrainingSizeIntoExecutionLimit(t *testing.T) {
	p, x, o := copyProfileFixture()
	p.LoadBatch = 4
	o.SkipTokens, o.KernelBytes, o.HBMBlocks = 0, 1024, []int64{1, 2, 3, 4}
	objects := []NativeTransferObject{o, o, o, o}
	for i := range objects {
		objects[i].Key = string(rune('a' + i))
	}
	plan := p.TransferPlan(NativeTransferShape{Direction: "load", Objects: objects}, x)
	if plan.GPUComplete != "kernel_0" || len(plan.CoverageGaps) != 2 {
		t.Fatal("native four-slot LOAD was split to fit two-object training data", plan)
	}
}

func TestNativeCopyProfileMixedLoadKeepsOneFourSlotBatchAndReportsCostGap(t *testing.T) {
	p, x, first := copyProfileFixture()
	p.LoadBatch = 4
	x.PoolMedia["warm"] = "dram"
	objects := []NativeTransferObject{first, first, first, first}
	for i := range objects {
		objects[i].Key = string(rune('a' + i))
		if i%2 == 0 {
			objects[i].Pool = "warm"
		}
		if i > 0 {
			objects[i].SkipTokens = 0
			objects[i].KernelBytes = 1024
			objects[i].HBMBlocks = []int64{1, 2, 3, 4}
		}
	}
	plan := p.TransferPlan(NativeTransferShape{Direction: "load", Objects: objects}, x)
	kernels, submits, replies := 0, 0, 0
	for _, stage := range plan.Stages {
		if stage.Name == "kernel_0" {
			kernels++
			if stage.DurationUS != 3840 {
				t.Fatal("combined scatter undercharged", stage)
			}
		}
		if stage.Name == "host_submit" {
			submits++
		}
		if stage.Name == "reply" {
			replies++
		}
	}
	if plan.GPUComplete != "kernel_0" || kernels != 1 || submits != 1 || replies != 1 {
		t.Fatal("pool boundary split the native operation/batch", plan)
	}
	if plan.StagingSlots != 4 || plan.StagingBufferBytes != 4096 {
		t.Fatal("native shared staging allocation not reported", plan)
	}
	if len(plan.CoverageGaps) == 0 {
		t.Fatal("unmeasured mixed-media composition silently accepted")
	}
}

func TestNativeCopyProfileRejectsUnpaidOrMalformedTransfer(t *testing.T) {
	for _, mutation := range []func(*NativeTransferExecution, *NativeTransferObject){
		func(x *NativeTransferExecution, o *NativeTransferObject) { x.HostSubmitUS = 0 },
		func(x *NativeTransferExecution, o *NativeTransferObject) { o.Segments = o.Segments[:1] },
		func(x *NativeTransferExecution, o *NativeTransferObject) { o.KernelBytes = 1024 },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("invalid transfer silently accepted")
				}
			}()
			p, x, o := copyProfileFixture()
			mutation(&x, &o)
			p.TransferPlan(NativeTransferShape{Direction: "load", Objects: []NativeTransferObject{o}}, x)
		}()
	}
}
