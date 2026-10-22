package kv

import (
	"fmt"
	"github.com/inference-sim/inference-sim/sim"
	"reflect"
	"strings"
	"testing"
)

type mixedTransferCost struct {
	nativePhaseTestCost
	shapes *[]sim.NativeTransferShape
}

func (c mixedTransferCost) TransferPlan(shape sim.NativeTransferShape) sim.NativeTransferPlan {
	*c.shapes = append(*c.shapes, shape)
	// Artificial per-object copies followed by one operation-level completion.
	stages := []sim.NativeStage{{Name: "submit", Resource: "cpu", Stream: "mp_host", DurationUS: 3}}
	previous := "submit"
	for i := range shape.Objects {
		name := fmt.Sprintf("object_%d", i)
		stages = append(stages, sim.NativeStage{Name: name, Resource: "copy", Stream: "native-kv-stream", DurationUS: 20, After: []string{previous}})
		previous = name
	}
	stages = append(stages, sim.NativeStage{Name: "reply", Resource: "cpu", Stream: "mp_reply", DurationUS: 2, After: []string{previous}})
	return sim.NativeTransferPlan{Stages: stages, ProducerWait: "object_0", GPUComplete: previous, ModelComplete: "submit", Published: "reply"}
}

func TestNativeMixedPoolOperationPreservesObjectOrderAndAtomicCompletion(t *testing.T) {
	for _, load := range []bool{false, true} {
		t.Run(map[bool]string{false: "store", true: "load_partial_prefix"}[load], func(t *testing.T) {
			h, s := hostChunkHarness(t, 2, 2)
			input := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}
			if err := s.ConfigureNativeHostGeometry(1, 64<<20); err != nil {
				t.Fatal(err)
			}
			if load {
				seedPeer(t, s, "seed", input)
				drainHost(t, h, s)
				first := s.hashes(input)[0]
				for _, block := range s.Blocks {
					if block.Hash != first {
						s.invalidate(block)
					}
				}
			}
			e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
				KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 24, BlockSizeTokens: 2}, BatchConfig: sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 16}}, s, nativeTestCost{})
			if err != nil {
				t.Fatal(err)
			}
			var shapes []sim.NativeTransferShape
			if err = e.EnableNativeAsync(2, mixedTransferCost{shapes: &shapes}); err != nil {
				t.Fatal(err)
			}
			checked := false
			e.EventObserver = func(event sim.Event, before bool) {
				x, ok := event.(*sim.NativeAsyncEvent)
				if !ok || before || x.Name != "native_phase_end" || !strings.HasSuffix(x.Phase, "/object_0") {
					return
				}
				checked = true
				if load {
					if s.readPending["mixed"] != 3 {
						t.Fatal("partial mixed LOAD published before the complete operation")
					}
				} else if s.fabric.Snapshot()["dram"]["ready"] != 0 || s.fabric.Snapshot()["cxl"]["ready"] != 0 {
					t.Fatal("mixed STORE published before all objects/reply")
				}
			}
			r := &sim.Request{ID: "mixed", InputTokens: input, OutputTokens: []sim.TokenID{10}, MaxOutputLen: 1}
			e.InjectArrival(r)
			e.Run()
			if !checked || len(shapes) != 1 || len(shapes[0].Objects) != 2 {
				t.Fatal("one native request operation was split by pool", len(shapes), shapes)
			}
			objects := shapes[0].Objects
			if !reflect.DeepEqual([]string{objects[0].Pool, objects[1].Pool}, []string{"dram", "cxl"}) {
				t.Fatal("native token/object order changed", objects)
			}
			if load && (objects[0].SkipTokens != 2 || objects[0].KernelBytes != 100 || objects[0].StagingBytes != 200 || objects[1].KernelBytes != 200) {
				t.Fatal("partial prefix changed whole-object staging geometry", objects)
			}
			if e.Metrics.CompletedRequests != 1 || s.UsedBlocks() != 0 || s.fabric.Pending() != 0 || s.nativeBatchPins != 0 {
				t.Fatal("mixed operation did not drain")
			}
		})
	}
}

type nativeTransferTestCost struct {
	nativePhaseTestCost
	modelUS int64
}

func (nativeTransferTestCost) ExecutionResources() []string {
	return []string{"cpu", "compute", "copy", "mp", "model_write"}
}
func (c nativeTransferTestCost) TransferPlan(shape sim.NativeTransferShape) sim.NativeTransferPlan {
	if shape.Direction != "store" || len(shape.Objects) != 1 || shape.Objects[0].Pool != "cxl" {
		panic("unexpected test geometry")
	}
	return sim.NativeTransferPlan{Stages: []sim.NativeStage{
		{Name: "submit", Resource: "mp", Stream: "mp_host", DurationUS: 3},
		{Name: "model_serial", Resource: "model_write", DurationUS: c.modelUS, After: []string{"submit"}},
		{Name: "model_latency", Delay: true, DurationUS: 7, After: []string{"model_serial"}},
		{Name: "gather", Resource: "compute", Stream: "native-kv-stream", DurationUS: 10, After: []string{"submit"}},
		{Name: "staging", Resource: "copy", Stream: "native-kv-stream", DurationUS: 20, After: []string{"gather"}},
		{Name: "reply", Resource: "mp", Stream: "mp_reply", DurationUS: 2, After: []string{"staging", "model_latency"}},
	}, ProducerWait: "gather", GPUComplete: "staging", ModelComplete: "model_latency", Published: "reply"}
}

func TestNativeTransferJoinsPhysicalAndModeledWorkAfterCancellation(t *testing.T) {
	for _, modelUS := range []int64{1, 500} {
		t.Run(fmt.Sprintf("model-%d", modelUS), func(t *testing.T) {
			h, s := hostChunkHarness(t, 2, 4)
			s.fabric.hostChunks.policy = OrderedHostPlacement{PoolOrder: []string{"cxl"}, EvictPools: []string{"cxl"}}
			if err := s.ConfigureNativeHostGeometry(1, 64<<20); err != nil {
				t.Fatal(err)
			}
			e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
				KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 24, BlockSizeTokens: 2}, BatchConfig: sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 16}}, s, nativeTestCost{})
			if err != nil {
				t.Fatal(err)
			}
			if err = e.EnableNativeAsync(2, nativeTransferTestCost{modelUS: modelUS}); err != nil {
				t.Fatal(err)
			}
			r := &sim.Request{ID: "cancel-join", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6}, MaxOutputLen: 1, Deadline: 10}
			gpu, model, reply := int64(-1), int64(-1), int64(-1)
			protected := false
			e.EventObserver = func(event sim.Event, before bool) {
				if x, ok := event.(*sim.NativeAsyncEvent); ok && !before && x.Name == "native_phase_end" {
					switch x.Phase {
					case "kv_1/staging":
						gpu = x.At
						if modelUS == 500 {
							protected = s.UsedBlocks() > 0 && s.fabric.Snapshot()["cxl"]["reserved"] == 2 && s.fabric.Snapshot()["cxl"]["ready"] == 0
						}
					case "kv_1/model_latency":
						model = x.At
					case "kv_1/reply":
						reply = x.At
					}
				}
			}
			e.InjectArrival(r)
			e.Run()
			if gpu < 0 || model < 0 || reply != max(gpu, model)+2 {
				t.Fatal("CUDA and model were added or not joined", gpu, model, reply)
			}
			if modelUS == 500 && (model <= gpu || !protected) {
				t.Fatal("source/publication protection vanished while model was pending")
			}
			if modelUS == 1 && model >= gpu {
				t.Fatal("model branch incorrectly waited on CUDA producer")
			}
			if r.TTFTSet || s.fabric.Pending() != 0 || s.UsedBlocks() != 0 || s.nativeBatchPins != 0 {
				t.Fatal("cancelled transfer did not drain")
			}
			cudaEnds, logicalEnds := 0, 0
			for _, record := range h.records {
				if record.Name == "transfer_cuda_complete" {
					cudaEnds++
				}
				if record.Name == "transfer_end" {
					logicalEnds++
				}
			}
			if cudaEnds != 1 || logicalEnds != 1 {
				t.Fatal("physical and logical boundaries were not independently emitted")
			}
		})
	}
}
