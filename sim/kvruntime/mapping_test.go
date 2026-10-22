package kvruntime

import "testing"

func TestHFScatterGatherPreservesHeadsPagesAndHoles(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"hbm": 1024 * 1024})
	r, _ := NewRuntime(e, m, 4)
	for _, x := range []struct {
		id    string
		bytes int64
	}{{"hf", 256 * 1024}, {"arena", 512 * 1024}, {"out", 256 * 1024}} {
		if _, err := m.Reserve(x.id, "hbm", "device", x.bytes); err != nil {
			t.Fatal(err)
		}
		_ = m.Allocate(x.id)
	}
	_ = m.Fill("hf", 17)
	_ = m.Fill("arena", 0)
	spans, err := HFScatterSpans(256)
	if err != nil {
		t.Fatal(err)
	}
	submitted, scatter, err := r.MapKernel("scatter", "cpu", "gpu", MappingKernel{Name: "index_copy", Source: "hf", Destination: "arena", Spans: spans, SubmitNS: 2, ServiceNS: 7})
	if err != nil {
		t.Fatal(err)
	}
	// Storage ownership begins at submission; readiness is checked at GPU use.
	if err := m.Free("hf"); err == nil {
		t.Fatal("mapping source freed after launch")
	}
	inverse := make([]Span, len(spans))
	for i, p := range spans {
		inverse[i] = Span{p.DestinationOffset, p.SourceOffset, p.Bytes}
	}
	_, _, err = r.MapKernel("gather", "cpu", "gpu", MappingKernel{Name: "gather", Source: "arena", Destination: "out", Spans: inverse, Parents: []int64{submitted}, StreamParent: scatter, SubmitNS: 20, ServiceNS: 7})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(10); err != nil {
		t.Fatal(err)
	}
	if m.Buffers["arena"].State != "ready" || m.Buffers["arena"].Holds != 1 || m.Buffers["arena"].Pins != 0 {
		t.Fatal("queued consumer lost lifetime/readiness distinction")
	}
	if err = m.Free("arena"); err == nil {
		t.Fatal("producer storage freed before queued consumer used it")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	for i, c := range m.Buffers["out"].Cells {
		if c != (Cell{17, int64(i) * 256, true}) {
			t.Fatal("head-major round trip corrupted")
		}
	}
	for i, c := range m.Buffers["arena"].Cells {
		if (i*256/16384)%2 == 1 && c.Origin != 0 {
			t.Fatal("physical hole overwritten")
		}
	}
}
