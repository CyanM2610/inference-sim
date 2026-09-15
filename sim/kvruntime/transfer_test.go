package kvruntime

import "testing"

func transferFixture(t *testing.T, depth int) (*Runtime, []Record) {
	t.Helper()
	e := NewEngine(nil)
	m, err := NewMemory(e, 16, map[string]int64{"hbm": 256, "dram": 256})
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range []struct{ id, pool string }{{"source", "hbm"}, {"target", "dram"}} {
		if _, err = m.Reserve(x.id, x.pool, "pinned", 128); err != nil {
			t.Fatal(err)
		}
		if err = m.Allocate(x.id); err != nil {
			t.Fatal(err)
		}
	}
	if err = m.Fill("source", 7); err != nil {
		t.Fatal(err)
	}
	r, err := NewRuntime(e, m, depth)
	if err != nil {
		t.Fatal(err)
	}
	return r, nil
}
func TestTransferFiniteQueueAndDataPublication(t *testing.T) {
	for _, depth := range []int{2, 64} {
		r, _ := transferFixture(t, depth)
		var spans []Span
		for i := int64(0); i < 8; i++ {
			spans = append(spans, Span{i * 16, (7 - i) * 16, 16})
		}
		v, err := r.Transfer(TransferSpec{ID: "store", Source: "source", Destination: "target", CPU: "cpu", DMA: "dma", Spans: spans, Cost: TransferCost{PlanNS: 3, SetupNS: 2, SubmitNS: 1, DMANS: 5, ObserveNS: 2}})
		if err != nil {
			t.Fatal(err)
		}
		if err = r.Engine.RunUntil(11); err != nil {
			t.Fatal(err)
		}
		if r.Memory.Buffers["source"].Pins != 1 || r.Memory.Buffers["target"].State != "writing" {
			t.Fatal("premature publication/release")
		}
		if err = r.Engine.Run(); err != nil {
			t.Fatal(err)
		}
		if v.CompleteNS != 48 || v.Bytes != 128 || v.Copies != 8 {
			t.Fatal(v)
		}
		wantCPU := int64(13)
		if depth == 2 {
			wantCPU = 37
		}
		if v.CPUFinishedNS != wantCPU {
			t.Fatalf("queue backpressure: got %d want %d", v.CPUFinishedNS, wantCPU)
		}
		for i, c := range r.Memory.Buffers["target"].Cells {
			if c != (Cell{7, int64(7-i) * 16, true}) {
				t.Fatal("reordered copy lost data")
			}
		}
		if r.Memory.Buffers["source"].Pins != 0 || r.Memory.Buffers["target"].State != "ready" || r.outstanding != 0 {
			t.Fatal("transfer lifecycle leak")
		}
	}
}
func TestNativeLayoutMatchesPhysicalAddresses(t *testing.T) {
	for _, layout := range []string{"packed", "layers", "paged"} {
		out, gpu, host, err := NativeSpans(layout, 128, false)
		if err != nil {
			t.Fatal(err)
		}
		back, _, _, err := NativeSpans(layout, 128, true)
		if err != nil {
			t.Fatal(err)
		}
		var total int64
		seen := map[int64]bool{}
		for i, p := range out {
			total += p.Bytes
			if p.SourceOffset+p.Bytes > gpu || p.DestinationOffset+p.Bytes > host {
				t.Fatal("layout exceeds buffer")
			}
			if back[i] != (Span{p.DestinationOffset, p.SourceOffset, p.Bytes}) {
				t.Fatal("restore did not reverse mapping")
			}
			if layout == "paged" {
				if seen[p.SourceOffset] {
					t.Fatal("alias")
				}
				seen[p.SourceOffset] = true
			}
		}
		if total != 117440512 {
			t.Fatal(total)
		}
		want := 1
		if layout == "layers" {
			want = 56
		}
		if layout == "paged" {
			want = 7168
		}
		if len(out) != want {
			t.Fatal(layout, len(out))
		}
	}
}

func TestHostReturnBeforeDMACompletionStillProtectsData(t *testing.T) {
	r, _ := transferFixture(t, 2)
	var spans []Span
	for i := int64(0); i < 8; i++ {
		spans = append(spans, Span{i * 16, i * 16, 16})
	}
	v, err := r.Transfer(TransferSpec{ID: "pageable", Source: "source", Destination: "target", CPU: "cpu", DMA: "dma", Spans: spans, Cost: TransferCost{PreparePerCopyNS: 2, SubmitNS: 1, DMANS: 5, ObserveNS: 2, WaitForDMA: true, ReturnLeadNS: 3}})
	if err != nil {
		t.Fatal(err)
	}
	if err = r.Engine.RunUntil(40); err != nil {
		t.Fatal(err)
	}
	if v.CPUFinishedNS != 40 || r.Memory.Buffers["target"].State != "writing" || r.Memory.Buffers["source"].Pins != 1 {
		t.Fatal("host return published unfinished DMA", v)
	}
	if err = r.Engine.Run(); err != nil {
		t.Fatal(err)
	}
	if v.CompleteNS != 45 || v.LastDMANS != 43 {
		t.Fatal(v)
	}
}
