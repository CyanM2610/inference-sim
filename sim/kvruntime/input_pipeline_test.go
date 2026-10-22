package kvruntime

import (
	"strings"
	"testing"
)

func inputFixture(t *testing.T) (*Runtime, *RequestInputs) {
	t.Helper()
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"gpu": 1 << 20, "host": 640})
	r, _ := NewRuntime(e, m, 8)
	p := InputPipelineProfile{GPUIndexBytes: 512, HostIndexBytes: 320, GPUAllocationQuantum: 512, ReplacedGPUInputBytes: 1536, Evidence: "test", Index: []InputIndexPoint{{Count: 1, PrepareNS: 5, Cost: TransferCost{SubmitNS: 2, DMANS: 3, HoldCPUUntilDMA: true}}}, NextInput: TransferCost{SubmitNS: 2, DMANS: 3, HoldCPUUntilDMA: true}}
	i, err := NewRequestInputs(r, p, "gpu", "host", []string{"page"}, []InputRequest{{ID: "req", Prompt: []uint64{7}, Output: []uint64{12345, 54321}}})
	if err != nil {
		t.Fatal(err)
	}
	m.ReserveGranular("token", "gpu", "GPU producer", 512, 8)
	m.Allocate("token")
	m.Fill("token", 12345)
	return r, i
}

func TestInputPipelineWaitsForIndexAndTokenCopies(t *testing.T) {
	r, i := inputFixture(t)
	e, m := r.Engine, r.Memory
	e.MustAdd(Task{Name: "index_copy_backlog", Resource: "copy_h2d", DurationNS: 1000})
	b, _, err := i.Bind("first", "req", 0, 1, []string{"page"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(100); err != nil {
		t.Fatal(err)
	}
	if b.indicesPinned || m.Buffers[i.indexGPU].Writers != 1 {
		t.Fatal("indices became readable before actual H2D completion")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if err = b.AcquireQuery(); err != nil {
		t.Fatal(err)
	}
	if err = b.ReleaseQuery(); err != nil {
		t.Fatal(err)
	}
	started := e.NowNS
	e.MustAdd(Task{Name: "next_input_backlog", Resource: "copy_d2d", DurationNS: 1000})
	if _, err = b.ObserveOutput("token", 12345, nil); err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(started + 100); err != nil {
		t.Fatal(err)
	}
	if i.requests["req"].produced != 0 || m.Buffers["token"].Pins != 1 {
		t.Fatal("token declared produced or source released before D2D completion")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if i.requests["req"].produced != 1 || m.Buffers[i.requests["req"].buffer].Cells[1] != (Cell{12345, 0, true}) {
		t.Fatal("next input value was not physically copied")
	}
	if err = b.ReleaseIndices(); err != nil {
		t.Fatal(err)
	}
	decode, _, err := i.Bind("decode", "req", 1, 1, []string{"page"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if err = decode.AcquireQuery(); err != nil {
		t.Fatal(err)
	}
	decode.ReleaseQuery()
	decode.ReleaseIndices()
	if err = m.Check(); err != nil {
		t.Fatal(err)
	}
}

func TestInputPipelineRejectsUnproducedOrCorruptDecodeInput(t *testing.T) {
	for _, corrupt := range []bool{false, true} {
		r, i := inputFixture(t)
		if corrupt {
			i.requests["req"].produced = 1
		} // readiness alone cannot certify actual input bytes.
		b, _, err := i.Bind("decode", "req", 1, 1, []string{"page"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err = r.Engine.Run(); err != nil {
			t.Fatal(err)
		}
		err = b.AcquireQuery()
		wanted := "before token production"
		if corrupt {
			wanted = "differs from independently"
		}
		if err == nil || !strings.Contains(err.Error(), wanted) {
			t.Fatalf("corrupt=%v, got %v", corrupt, err)
		}
	}
}
