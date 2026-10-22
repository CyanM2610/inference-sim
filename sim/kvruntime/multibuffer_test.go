package kvruntime

import "testing"

func TestMultiBufferPlanFailureIsAtomic(t *testing.T) {
	r, _ := transferFixture(t, 4)
	m := r.Memory
	_, _ = m.Reserve("unready", "hbm", "device", 128)
	_ = m.Allocate("unready")
	_, _ = m.Reserve("second", "dram", "pinned", 128)
	_ = m.Allocate("second")
	_, err := r.Transfer(TransferSpec{ID: "list", CPU: "cpu", DMA: "dma", Spans: []Span{{0, 0, 128}, {0, 0, 128}}, Endpoints: []CopyEndpoints{{"source", "target"}, {"unready", "second"}}, Cost: TransferCost{SubmitNS: 1, DMANS: 2}})
	if err == nil {
		err = r.Engine.Run()
	}
	if err == nil {
		t.Fatal("unpublished source accepted")
	}
	if m.Buffers["source"].Pins != 0 || m.Buffers["target"].Writers != 0 {
		t.Fatal("partial pin/writer leaked on rejected plan")
	}
}
func TestMultiBufferTransferPublishesAllTargetsTogether(t *testing.T) {
	r, _ := transferFixture(t, 4)
	m := r.Memory
	_, _ = m.Reserve("second", "dram", "pinned", 128)
	_ = m.Allocate("second")
	tx, err := r.Transfer(TransferSpec{ID: "list", CPU: "cpu", DMA: "dma", Spans: []Span{{0, 0, 64}, {64, 64, 64}}, Endpoints: []CopyEndpoints{{"source", "target"}, {"source", "second"}}, Cost: TransferCost{SubmitNS: 1, DMANS: 5, ObserveNS: 3}})
	if err != nil {
		t.Fatal(err)
	}
	if err = r.Engine.RunUntil(7); err != nil {
		t.Fatal(err)
	}
	if m.Buffers["source"].Pins != 1 || m.Buffers["target"].State != "writing" || m.Buffers["second"].State != "writing" {
		t.Fatal("target exposed after only one DMA")
	}
	if err = r.Engine.Run(); err != nil {
		t.Fatal(err)
	}
	if tx.Bytes != 128 || m.Buffers["source"].Pins != 0 || m.Buffers["second"].Cells[4].Offset != 64 {
		t.Fatal("bad tensor list result")
	}
}
