package kvruntime

import "testing"

func TestScalarCopySharesPhysicalLedgerWithoutRoundingPayload(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"hbm": 1536, "dram": 512})
	if _, err := m.Reserve("kv", "hbm", "kv", 1024); err != nil {
		t.Fatal(err)
	}
	for _, v := range []struct{ id, pool string }{{"token", "hbm"}, {"host", "dram"}} {
		if _, err := m.ReserveGranular(v.id, v.pool, "scalar_storage", 512, 8); err != nil {
			t.Fatal(err)
		}
		if err := m.Allocate(v.id); err != nil {
			t.Fatal(err)
		}
	}
	if err := m.BeginWrite("token"); err != nil {
		t.Fatal(err)
	}
	if err := m.ComputeRange("token", 0, 8, 71, 0); err != nil {
		t.Fatal(err)
	}
	if err := m.Publish("token"); err != nil {
		t.Fatal(err)
	}
	r, _ := NewRuntime(e, m, 4)
	tx, err := r.Transfer(TransferSpec{ID: "observe_token", Source: "token", Destination: "host", CPU: "request", DMA: "pcie", Spans: []Span{{0, 0, 8}}, Cost: TransferCost{SubmitNS: 2, DMANS: 3}})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if tx.Bytes != 8 || len(m.Buffers["kv"].Cells) != 4 || m.Used["hbm"] != 1536 || m.Used["dram"] != 512 {
		t.Fatal("scalar payload or allocator footprint inflated")
	}
	if m.Buffers["host"].Cells[0] != (Cell{71, 0, true}) || m.Buffers["host"].Cells[1].Valid {
		t.Fatal("scalar copy changed uninitialized storage padding")
	}
	if err = m.Check(); err != nil {
		t.Fatal(err)
	}
}

func TestTransferRejectsImplicitGranularityConversionBeforePin(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"hbm": 512, "dram": 512})
	m.Reserve("kv", "hbm", "kv", 512)
	m.Allocate("kv")
	m.Fill("kv", 1)
	m.ReserveGranular("scalar", "dram", "pinned", 512, 8)
	m.Allocate("scalar")
	r, _ := NewRuntime(e, m, 4)
	_, err := r.Transfer(TransferSpec{ID: "bad", Source: "kv", Destination: "scalar", CPU: "cpu", DMA: "dma", Spans: []Span{{0, 0, 256}}, Cost: TransferCost{DMANS: 1}})
	if err == nil || m.Buffers["kv"].Pins != 0 || m.Buffers["scalar"].Writers != 0 {
		t.Fatal("implicit granularity conversion or partial mutation accepted")
	}
}
