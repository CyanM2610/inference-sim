package kvruntime

import "testing"

func TestHostWaitHoldsCPUAcrossDMAQueueAndRateChanges(t *testing.T) {
	e := NewEngine(nil)
	var returned, other int64
	backlog := e.MustAdd(Task{Name: "backlog", Resource: "dma", DurationNS: 100})
	device := e.MustAdd(Task{Name: "device_copy", Resource: "dma", ServiceClass: "copy", DurationNS: 10, Parents: []int64{backlog}})
	e.MustAdd(Task{Name: "stream_sync", Resource: "cpu", WaitFor: []int64{device}, Finish: func() error { returned = e.NowNS; return nil }})
	_ = e.At(5, func() {
		e.MustAdd(Task{Name: "same_thread_next_call", Resource: "cpu", DurationNS: 1, Finish: func() error { other = e.NowNS; return nil }})
	})
	_ = e.At(105, func() {
		if err := e.SetServiceRate("copy", 500000); err != nil {
			t.Fatal(err)
		}
	})
	if err := e.RunUntil(110); err != nil {
		t.Fatal(err)
	}
	if returned != 0 || other != 0 {
		t.Fatal("CPU released at predicted rather than actual DMA completion")
	}
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if returned != 115 || other != 116 {
		t.Fatalf("wrong host wait/continuation %d %d", returned, other)
	}
}

func TestHostWaitForAlreadyCompletedTargetIsImmediate(t *testing.T) {
	e := NewEngine(nil)
	target := e.MustAdd(Task{Name: "done", Resource: "dma", DurationNS: 2})
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	var at int64
	e.MustAdd(Task{Name: "observe", Resource: "cpu", WaitFor: []int64{target}, Finish: func() error { at = e.NowNS; return nil }})
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if at != 2 {
		t.Fatal("already-completed wait consumed fictional time")
	}
}
