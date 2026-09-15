package kvruntime

import (
	"reflect"
	"testing"
)

func TestDependencyResourcesPreserveNanoseconds(t *testing.T) {
	var events []Record
	e := NewEngine(func(r Record) { events = append(events, r) })
	a := e.MustAdd(Task{Name: "submit", Resource: "cpu", DurationNS: 1450})
	b := e.MustAdd(Task{Name: "dma", Resource: "dma", DurationNS: 2251, Parents: []int64{a}})
	e.MustAdd(Task{Name: "independent_cpu", Resource: "cpu", DurationNS: 5000, Parents: []int64{a}})
	e.MustAdd(Task{Name: "publish", Resource: "cpu", DurationNS: 137, Parents: []int64{b}})
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if e.NowNS != 6587 {
		t.Fatalf("resources/precision lost: %d", e.NowNS)
	}
	var dmaStart, dmaEnd int64
	for _, r := range events {
		if r.Reason == "dma" {
			if r.Name == "task_start" {
				dmaStart = r.TimeNS
			}
			if r.Name == "task_end" {
				dmaEnd = r.TimeNS
			}
		}
	}
	if dmaStart != 1450 || dmaEnd != 3701 {
		t.Fatal("CPU/DMA did not overlap correctly")
	}
}
func TestBoundedReadyHeadDoesNotLetHostThreadOvertake(t *testing.T) {
	e := NewEngine(nil)
	credit := false
	var order []string
	e.MustAdd(Task{Name: "blocked_api", Resource: "cpu", DurationNS: 1, Ready: func() bool { return credit }, Finish: func() error { order = append(order, "api"); return nil }})
	e.MustAdd(Task{Name: "next_host_instruction", Resource: "cpu", DurationNS: 1, Finish: func() error { order = append(order, "next"); return nil }})
	e.MustAdd(Task{Name: "device_releases_credit", Resource: "dma", DurationNS: 7, Finish: func() error { credit = true; return nil }})
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []string{"api", "next"}) || e.NowNS != 9 {
		t.Fatal(order, e.NowNS)
	}
}
func TestMissingProgressIsError(t *testing.T) {
	e := NewEngine(nil)
	e.MustAdd(Task{Name: "blocked", Resource: "cpu", Ready: func() bool { return false }})
	if e.Run() == nil {
		t.Fatal("deadlock reported success")
	}
	if _, err := e.Add(Task{Parents: []int64{999}}); err == nil {
		t.Fatal("unknown dependency accepted")
	}
}
func TestSymbolicCopyEnforcesOwnershipAndContent(t *testing.T) {
	e := NewEngine(nil)
	m, err := NewMemory(e, 16, map[string]int64{"hbm": 128, "dram": 64})
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range []struct{ id, pool string }{{"a", "hbm"}, {"b", "dram"}} {
		if _, err = m.Reserve(x.id, x.pool, "pinned", 64); err != nil {
			t.Fatal(err)
		}
		if err = m.Allocate(x.id); err != nil {
			t.Fatal(err)
		}
	}
	if m.Pin("a") == nil {
		t.Fatal("uninitialized memory visible")
	}
	if err = m.Fill("a", 42); err != nil {
		t.Fatal(err)
	}
	if err = m.Pin("a"); err != nil {
		t.Fatal(err)
	}
	if m.Free("a") == nil {
		t.Fatal("in-flight source freed")
	}
	if err = m.BeginWrite("b"); err != nil {
		t.Fatal(err)
	}
	if m.Pin("b") == nil {
		t.Fatal("partial destination published")
	}
	if err = m.Copy("a", "b", 32, 0, 32); err != nil {
		t.Fatal(err)
	}
	if err = m.Copy("a", "b", 0, 32, 32); err != nil {
		t.Fatal(err)
	}
	if m.Copy("a", "b", 64, 0, 16) == nil {
		t.Fatal("out of bounds accepted")
	}
	if err = m.Publish("b"); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m.Buffers["b"].Cells, []Cell{{42, 32, true}, {42, 48, true}, {42, 0, true}, {42, 16, true}}) {
		t.Fatal("content/offset corrupted")
	}
	if err = m.Unpin("a"); err != nil {
		t.Fatal(err)
	}
	if err = m.Register("b"); err != nil {
		t.Fatal(err)
	}
	if m.Free("b") == nil {
		t.Fatal("registered host pages freed")
	}
	if err = m.Unregister("b"); err != nil {
		t.Fatal(err)
	}
	if err = m.Free("a"); err != nil {
		t.Fatal(err)
	}
	if err = m.Free("b"); err != nil {
		t.Fatal(err)
	}
	if err = m.Check(); err != nil {
		t.Fatal(err)
	}
	if m.Used["hbm"] != 0 || m.Used["dram"] != 0 {
		t.Fatal("capacity leak")
	}
}
