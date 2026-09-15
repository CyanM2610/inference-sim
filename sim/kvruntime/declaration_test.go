package kvruntime

import "testing"

func TestDeclaredEndpointsAllocateAtProducerTime(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"hbm": 512, "dram": 8})
	if _, err := m.Declare("token", "hbm", "scalar", 512, 8); err != nil {
		t.Fatal(err)
	}
	if m.Used["hbm"] != 0 || len(m.Buffers["token"].Cells) != 0 {
		t.Fatal("future endpoint charged early")
	}
	if m.Pin("token") == nil || m.Hold("token") == nil || m.BeginWrite("token") == nil {
		t.Fatal("future endpoint was usable")
	}
	m.ReserveGranular("host", "dram", "pinned", 8, 8)
	m.Allocate("host")
	r, _ := NewRuntime(e, m, 4)
	producer := e.MustAdd(Task{Name: "argmax", Resource: "gpu", DurationNS: 10, Finish: func() error {
		if err := m.Allocate("token"); err != nil {
			return err
		}
		if err := m.BeginWrite("token"); err != nil {
			return err
		}
		if err := m.ComputeRange("token", 0, 8, 19, 0); err != nil {
			return err
		}
		return m.Publish("token")
	}})
	tx, err := r.Transfer(TransferSpec{ID: "output", Source: "token", Destination: "host", CPU: "cpu", DMA: "dma", Parents: []int64{producer}, Spans: []Span{{0, 0, 8}}, Cost: TransferCost{DMANS: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(9); err != nil {
		t.Fatal(err)
	}
	if m.Used["hbm"] != 0 {
		t.Fatal("declaration consumed memory before producer")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if tx.CompleteNS != 12 || m.Used["hbm"] != 512 || m.Buffers["host"].Cells[0].Origin != 19 {
		t.Fatal("deferred allocation/copy failed")
	}
	if err = m.Check(); err != nil {
		t.Fatal(err)
	}
}

func TestDeclaredAllocationStillEnforcesActualCapacity(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 8, map[string]int64{"hbm": 8})
	m.Declare("future", "hbm", "scalar", 8, 8)
	m.Reserve("active", "hbm", "scalar", 8)
	if m.Allocate("future") == nil || m.Buffers["future"].State != "declared" || m.Used["hbm"] != 8 {
		t.Fatal("declaration bypassed physical capacity")
	}
	if err := m.Free("future"); err != nil {
		t.Fatal(err)
	}
	if m.Used["hbm"] != 8 {
		t.Fatal("cancelling declaration freed another buffer's capacity")
	}
}

func TestDeclaredTargetPlanFailsWithoutStrandingSourcePin(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 8, map[string]int64{"hbm": 8, "dram": 8})
	m.Reserve("source", "hbm", "scalar", 8)
	m.Allocate("source")
	m.BeginWrite("source")
	m.ComputeRange("source", 0, 8, 7, 0)
	m.Publish("source")
	m.Declare("future", "dram", "scalar", 8, 8)
	r, _ := NewRuntime(e, m, 4)
	_, err := r.Transfer(TransferSpec{ID: "early", Source: "source", Destination: "future", CPU: "cpu", DMA: "dma", Spans: []Span{{0, 0, 8}}, Cost: TransferCost{DMANS: 1}})
	if err == nil || m.Buffers["source"].Pins != 0 || m.Used["dram"] != 0 {
		t.Fatal("unallocated destination was accepted or failed plan stranded source")
	}
}
