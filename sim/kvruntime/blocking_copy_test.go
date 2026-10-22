package kvruntime

import "testing"

func TestBlockingAPIOwnsHostUntilDMACompletion(t *testing.T) {
	r, _ := transferFixture(t, 8)
	e, m := r.Engine, r.Memory
	v, err := r.Transfer(TransferSpec{ID: "blocking", Source: "source", Destination: "target", CPU: "cpu", DMA: "dma", Spans: []Span{{Bytes: 128}}, Cost: TransferCost{SubmitNS: 5, DMANS: 100, ObserveNS: 2, BlockingAPI: true}})
	if err != nil {
		t.Fatal(err)
	}
	var competingStart int64
	_ = e.At(10, func() {
		e.MustAdd(Task{Name: "same_thread_work", Resource: "cpu", DurationNS: 1, Start: func() error { competingStart = e.NowNS; return nil }})
	})
	if err = e.RunUntil(50); err != nil {
		t.Fatal(err)
	}
	if competingStart != 0 {
		t.Fatalf("blocking API released its host thread while DMA was running: competing start=%d", competingStart)
	}
	if m.Buffers["source"].Pins != 1 || m.Buffers["target"].Writers != 1 {
		t.Fatal("blocking wait released physical ownership")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if competingStart != 105 || v.CPUFinishedNS != 105 || v.LastDMANS != 105 || v.CompleteNS != 108 {
		t.Fatalf("wrong host/DMA completion ordering: competing=%d result=%+v", competingStart, v)
	}
	if err = m.Check(); err != nil {
		t.Fatal(err)
	}
}
