package kvruntime

import "testing"

func TestPhysicalGatherPreservesUndefinedTailAndStrictConsumerRejectsIt(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"hbm": 16384 * 4})
	r, _ := NewRuntime(e, m, 4)
	for _, id := range []string{"page", "gather", "valid", "invalid"} {
		_, _ = m.Reserve(id, "hbm", "device", 16384)
		_ = m.Allocate(id)
	}
	_ = m.BeginWrite("page")
	_ = m.ComputeRange("page", 0, 256, 1, 0)
	_ = m.Publish("page")
	_, first, err := r.MapKernel("select", "cpu", "gpu", MappingKernel{Name: "index_select", Source: "page", Destination: "gather", Spans: []Span{{0, 0, 16384}}, AllowUndefined: true, SubmitNS: 1, ServiceNS: 2})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = r.MapKernel("trim", "cpu", "gpu", MappingKernel{Name: "trim", Source: "gather", Destination: "valid", Spans: []Span{{0, 0, 256}}, Parents: []int64{first}, SubmitNS: 1, ServiceNS: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if m.Buffers["gather"].Cells[1].Valid || !m.Buffers["valid"].Cells[0].Valid {
		t.Fatal("unknown padding was converted to meaningful KV")
	}
	_, _, err = r.MapKernel("bad_read", "cpu", "gpu", MappingKernel{Name: "bad_read", Source: "gather", Destination: "invalid", Spans: []Span{{256, 0, 256}}, SubmitNS: 1, ServiceNS: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err == nil {
		t.Fatal("undefined KV consumed by strict copy")
	}
}

func TestCompoundWriterCollectsMultipleSourcesWithoutEarlyPublication(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 16, map[string]int64{"hbm": 64})
	r, _ := NewRuntime(e, m, 4)
	for _, x := range []struct {
		id    string
		bytes int64
	}{{"a", 16}, {"b", 16}, {"out", 32}} {
		_, _ = m.Reserve(x.id, "hbm", "device", x.bytes)
		_ = m.Allocate(x.id)
	}
	_ = m.Fill("a", 1)
	_ = m.Fill("b", 2)
	_ = m.BeginWriteOwned("out", "batch")
	_, last, err := r.MapKernel("map", "cpu", "gpu", MappingKernel{Name: "scatter", Spans: []Span{{0, 0, 16}, {0, 16, 16}}, Endpoints: []CopyEndpoints{{"a", "out"}, {"b", "out"}}, WriteOwner: "batch", SubmitNS: 2, ServiceNS: 3})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if m.Buffers["out"].State != "writing" {
		t.Fatal("kernel published a compound destination")
	}
	if err = m.Publish("out"); err == nil {
		t.Fatal("foreign writer published compound output")
	}
	e.MustAdd(Task{Name: "publish", Operation: "batch", Resource: "cpu", Parents: []int64{last}, Finish: func() error { return m.PublishOwned("out", "batch") }})
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if m.Buffers["out"].Cells[0].Origin != 1 || m.Buffers["out"].Cells[1].Origin != 2 {
		t.Fatal("multi-source mapping lost content")
	}
	if err = m.Invalidate("out"); err != nil {
		t.Fatal(err)
	}
	if m.Buffers["out"].Cells[0].Valid || m.Used["hbm"] != 64 {
		t.Fatal("reuse changed physical allocation or retained old validity")
	}
}
