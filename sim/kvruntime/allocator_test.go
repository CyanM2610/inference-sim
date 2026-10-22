package kvruntime

import "testing"

func TestWarmAllocatorBackingIsNotAddedToLiveLeases(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 16, map[string]int64{"gpu": 1024})
	if err := m.RetainAllocator("gpu", 512, true); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Reserve("tensor", "gpu", "device", 256); err != nil {
		t.Fatal(err)
	}
	if m.Used["gpu"] != 256 || m.PhysicalUsed("gpu") != 512 {
		t.Fatal("live lease double-counted reserved backing")
	}
	if err := m.Free("tensor"); err != nil {
		t.Fatal(err)
	}
	if m.Used["gpu"] != 0 || m.PhysicalUsed("gpu") != 512 {
		t.Fatal("cached backing disappeared on tensor release")
	}
	if _, err := m.Reserve("cold_growth", "gpu", "device", 768); err == nil {
		t.Fatal("unmeasured cold allocator growth accepted")
	}
	if err := m.Check(); err != nil {
		t.Fatal(err)
	}
}
