package kvruntime

import "testing"

func TestTorchArenaDirectionsPreserveUnusedSlots(t *testing.T) {
	p := &Profile{QueueDepth: 8, TorchCopies: &TorchCopyProfile{GPUSlots: 40, HostSlots: 64, BaselineAllocatedBytes: 15 << 30, ReservedBytes: 16 << 30}}
	for _, direction := range []string{"d2h", "h2d"} {
		x := TransferPoint{Blocks: 16, Direction: direction, Layout: "paged", HostMemory: "pinned", Copies: 896, Cost: TransferCost{SubmitNS: 2, DMANS: 5}}
		r, err := RunTorchCopyCase(p, x, nil)
		if err != nil {
			t.Fatal(err)
		}
		if r.Transfer.Bytes != 16*917504 || r.Transfer.Copies != 896 {
			t.Fatalf("wrong physical transfer: %+v", r.Transfer)
		}
		if r.FinalActive["hbm"] != 40*917504 || r.FinalActive["dram"] != 64*917504 {
			t.Fatal("physical arena shape changed")
		}
		want := int64(56 * 64 * 64)
		if direction == "h2d" {
			want = 56 * 40 * 64
		}
		if r.CellsVerified != want {
			t.Fatalf("did not verify untouched destination slots: %d", r.CellsVerified)
		}
	}
}
