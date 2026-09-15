package kvruntime

import "testing"

func TestCopyStreamDrainAndYieldChangeScalarObservation(t *testing.T) {
	for _, policy := range []string{"fifo", "stream_drain"} {
		e := NewEngine(nil)
		if err := e.ConfigureResource("copy", policy); err != nil {
			t.Fatal(err)
		}
		var observed int64
		first := e.MustAdd(Task{Name: "kv1", Resource: "copy", ResourceGroup: "kv_stream", DurationNS: 10})
		e.MustAdd(Task{Name: "kv2", Resource: "copy", ResourceGroup: "kv_stream", DurationNS: 10, Parents: []int64{first}})
		_ = e.At(5, func() {
			e.MustAdd(Task{Name: "token", Resource: "copy", ResourceGroup: "request_stream", DurationNS: 1, Finish: func() error { observed = e.NowNS; return nil }})
		})
		if err := e.Run(); err != nil {
			t.Fatal(err)
		}
		want := int64(11)
		if policy == "stream_drain" {
			want = 21
		}
		if observed != want {
			t.Fatalf("%s observed at %d want %d", policy, observed, want)
		}
	}
	e := NewEngine(nil)
	_ = e.ConfigureResource("copy", "stream_drain")
	var observed int64
	first := e.MustAdd(Task{Name: "kv1", Resource: "copy", ResourceGroup: "kv_stream", DurationNS: 10})
	gap := e.MustAdd(Task{Name: "host_completion_and_next_submit", Resource: "host", DurationNS: 4, Parents: []int64{first}})
	e.MustAdd(Task{Name: "kv2", Resource: "copy", ResourceGroup: "kv_stream", DurationNS: 10, Parents: []int64{gap}})
	_ = e.At(5, func() {
		e.MustAdd(Task{Name: "token", Resource: "copy", ResourceGroup: "request_stream", DurationNS: 1, Finish: func() error { observed = e.NowNS; return nil }})
	})
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if observed != 11 {
		t.Fatal("stream retained device while no command was ready")
	}
}
