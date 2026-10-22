package kv

import (
	"fmt"
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

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
