package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

// NativePeerRuntime binds physical PeerCache slots to the fine event runtime.
// Arenas are already allocated at admission (steady state). Slot reuse changes
// contents, never allocated byte count. Only single-instance pinned DRAM works.
type NativePeerRuntime struct {
	internalOutputReferences    map[string]map[string]uint64
	pagedPhaseObserver          func(string, bool) error
	pagedServiceClasses         kvruntime.PagedServiceClasses
	hbmSlots, hostSlots         int64
	inputs                      *kvruntime.RequestInputs
	outputEnabled               bool
	pagedEnabled, forwardActive bool
	pagedContents               map[string][]kvruntime.Cell
	origins                     map[string]uint64
	forwardSerial               int64
	Runtime                     *kvruntime.Runtime
	fabric                      *PeerFabric
	profile                     *kvruntime.Profile
	points                      map[string]kvruntime.TransferPoint
	hostFree                    []string
	hostLeased                  map[string]bool
	scheduled                   map[int64]bool
	completed                   []func()
	sink                        func(kvruntime.Record)
}

// NativeForward schedules the measured batch-one envelope as a dependent
// layer DAG. Its microsecond return value is rounded only at the BLIS boundary.
func (s *PeerCache) NativeForward(req string, prefix, query int64) (int64, error) {
	n := s.fabric.native
	if n == nil {
		return 0, fmt.Errorf("native runtime absent")
	}
	var point *kvruntime.ComputeShape
	for i := range n.profile.ComputeGrid {
		p := &n.profile.ComputeGrid[i]
		if p.Prefix == prefix && p.Query == query {
			point = p
			break
		}
	}
	if point == nil {
		return 0, fmt.Errorf("unmeasured forward prefix=%d query=%d; no extrapolation", prefix, query)
	}
	var weights []float64
	for _, c := range n.profile.Compute {
		if c.Length == 256 && c.Mode == "resident" {
			index := 0
			if query == 1 {
				index = 1
			}
			if len(c.ForwardWeights) > index {
				weights = c.ForwardWeights[index]
			}
			break
		}
	}
	if len(weights) != 30 {
		return 0, fmt.Errorf("missing forward structural calibration")
	}
	n.advance(s.clock)
	e := n.Runtime.Engine
	n.forwardSerial++
	operation := fmt.Sprintf("forward/%s/%d", req, n.forwardSerial)
	e.Emit(kvruntime.Record{Name: "forward_shape", Operation: operation, PrefixTokens: prefix, QueryTokens: query, DurationNS: point.WallNS, Reason: "batch_one_exact_shape_wall; layer_fractions_from_256_control; GPU_resident_input; transient_allocator_peak_recorded_not_placed"})
	var tail, assigned int64
	for i, w := range weights {
		ns := int64(float64(point.WallNS) * w)
		if i == 29 {
			ns = point.WallNS - assigned
		}
		assigned += ns
		var parents []int64
		if tail != 0 {
			parents = []int64{tail}
		}
		name := fmt.Sprintf("forward_segment_%02d", i)
		tail = e.MustAdd(kvruntime.Task{Name: name, Operation: operation, Resource: "forward_envelope", DurationNS: ns, Parents: parents})
	}
	n.arm(s.schedule)
	return point.WallNS, e.Error()
}

func (f *PeerFabric) ConfigureNative(p *kvruntime.Profile, numa int, sink func(kvruntime.Record)) error {
	if f.native != nil || f.external != nil || len(f.stores) != 1 || len(f.pools) != 1 || f.pools["dram"] == nil || f.bytes != 917504 || f.stores[0].BlockSizeTokens != 16 || f.active != 0 || f.serial != 0 {
		return fmt.Errorf("native backend requires one idle 7B-GQA instance, 16-token blocks and one DRAM pool")
	}
	if f.mechanisms.CompletionUS != 0 {
		return fmt.Errorf("native completion observation is calibrated; separate completion_us would double count")
	}
	s := f.stores[0]
	if len(s.access) != 1 || s.access[0].Pool != "dram" || len(s.access[0].ReadPath) != 1 || len(s.access[0].WritePath) != 1 || s.access[0].ReadPath[0] != s.access[0].WritePath[0] {
		return fmt.Errorf("native backend requires one shared transfer gate for both directions")
	}
	n, err := newNativePeerRuntime(p, numa, s.TotalBlocks, f.pools["dram"].config.CapacityBlocks, sink)
	if err != nil {
		return err
	}
	n.fabric = f
	f.native = n
	return nil
}

// newNativePeerRuntime binds physical arenas without attaching a policy clock.
// Both the legacy BLIS pump and the externally acknowledged controller use it.
func newNativePeerRuntime(p *kvruntime.Profile, numa int, hbmSlots, hostSlots int64, sink func(kvruntime.Record)) (*NativePeerRuntime, error) {
	if p == nil || hbmSlots <= 0 || hostSlots <= 0 {
		return nil, fmt.Errorf("native arenas require a profile and positive capacities")
	}
	n := &NativePeerRuntime{hbmSlots: hbmSlots, hostSlots: hostSlots, origins: map[string]uint64{}, profile: p, points: map[string]kvruntime.TransferPoint{}, hostLeased: map[string]bool{}, scheduled: map[int64]bool{}, sink: sink}
	for _, direction := range []string{"d2h", "h2d"} {
		point, err := p.Find(numa, "paged", "pinned", 1, direction)
		if err != nil {
			return nil, err
		}
		n.points[direction] = point
	}
	e := kvruntime.NewEngine(sink)
	const blockBytes = int64(917504)
	m, err := kvruntime.NewMemory(e, 256, map[string]int64{"hbm": hbmSlots * blockBytes, "dram": hostSlots * blockBytes})
	if err != nil {
		return nil, err
	}
	n.Runtime, err = kvruntime.NewRuntime(e, m, p.QueueDepth)
	if err != nil {
		return nil, err
	}
	for _, pool := range []string{"hbm", "dram"} {
		capacity := hbmSlots
		if pool == "dram" {
			capacity = hostSlots
		}
		for slot := int64(0); slot < capacity; slot++ {
			id := fmt.Sprintf("%s/slot/%d", pool, slot)
			kind := "device_arena_slot"
			if pool == "dram" {
				kind = "pinned_arena_slot"
				n.hostFree = append(n.hostFree, id)
			}
			if _, err := m.Reserve(id, pool, kind, blockBytes); err != nil {
				return nil, err
			}
			if err := m.Allocate(id); err != nil {
				return nil, err
			}
			if err := m.Fill(id, 0); err != nil {
				return nil, err
			}
			e.Emit(kvruntime.Record{Name: "arena_slot_bind", Destination: id, Resource: pool, Bytes: blockBytes, SourceOffset: slot * 16384, PhysicalLayerStride: capacity * 16384, Reason: "56_layer_major_slices_of_16KiB"})
		}
	}
	e.Emit(kvruntime.Record{Name: "native_scope", Reason: "warm preallocated arenas; 56x16KiB per-block native service; one transaction gate; BLIS compute remains its configured model; arena setup, CPU/compute contention and coalesced cross-block transfer are not calibrated"})
	return n, nil
}
func (n *NativePeerRuntime) hbmID(id int64) string { return fmt.Sprintf("hbm/slot/%d", id) }
func (n *NativePeerRuntime) reserveEntry(pool string) string {
	if pool != "dram" || len(n.hostFree) == 0 {
		panic("native arena capacity disagrees with peer pool")
	}
	id := n.hostFree[0]
	n.hostFree = n.hostFree[1:]
	n.hostLeased[id] = true
	return id
}
func (n *NativePeerRuntime) releaseEntry(entry *peerEntry) {
	id := entry.nativeBuffer
	b := n.Runtime.Memory.Buffers[id]
	if !n.hostLeased[id] || b == nil || b.Pins != 0 || b.Writers != 0 || b.Holds != 0 {
		panic("native L2 slot recycled while active")
	}
	delete(n.hostLeased, id)
	n.hostFree = append(n.hostFree, id)
}
func (n *NativePeerRuntime) advance(now int64) {
	e := n.Runtime.Engine
	at := now * 1000
	if now < 0 || now > math.MaxInt64/1000 {
		panic("native timestamp overflow")
	}
	if err := e.At(at, func() {}); err != nil {
		panic(err)
	}
	if err := e.RunUntil(at); err != nil {
		panic(err)
	}
}
func (n *NativePeerRuntime) computed(now, id int64, hash string) {
	n.advance(now)
	if n.pagedEnabled {
		if err := n.checkPagedContents(hash, n.hbmID(id)); err != nil {
			panic(err)
		}
		return
	}
	m := n.Runtime.Memory
	buffer := n.hbmID(id)
	if err := m.BeginWrite(buffer); err != nil {
		panic(err)
	}
	origin := n.origins[hash]
	if origin == 0 {
		origin = uint64(len(n.origins)) + 1
		n.origins[hash] = origin
		n.Runtime.Engine.Emit(kvruntime.Record{Name: "kv_content_identity", Origin: origin, PayloadKey: hash, Reason: "collision_free_session_identity_for_full_prefix_hash"})
	}
	if err := m.ComputeRange(buffer, 0, n.fabric.bytes, origin, 0); err != nil {
		panic(err)
	}
	if err := m.Publish(buffer); err != nil {
		panic(err)
	}
}
func (n *NativePeerRuntime) submit(now int64, j *peerJob) {
	n.advance(now)
	direction := "d2h"
	if j.record.Destination == "hbm" {
		direction = "h2d"
	}
	point := n.points[direction]
	spans := make([]kvruntime.Span, 56)
	// Logical buffer per physical block permits disjoint blocks to carry
	// independent pins. Its 56 slices map to layer-major arena addresses.
	for i := range spans {
		spans[i] = kvruntime.Span{SourceOffset: int64(i) * 16384, DestinationOffset: int64(i) * 16384, Bytes: 16384}
	}
	dma, group := "pcie_copy", ""
	if n.outputEnabled {
		dma = "copy_" + direction
		group = "kv_stream"
	}
	_, err := n.Runtime.Transfer(kvruntime.TransferSpec{ID: fmt.Sprintf("peer/%d/%s", j.record.Transaction, j.record.Reason), Source: j.nativeSource(), Destination: j.nativeDestination(), CPU: "copy_thread", DMA: dma, StreamGroup: group, Spans: spans, Cost: point.Cost, Done: func(tx *kvruntime.TransferResult) {
		at := (tx.CompleteNS + 999) / 1000
		n.completed = append(n.completed, func() {
			n.fabric.emit(PeerRecord{Time: at, Name: "native_handoff", Transaction: j.record.Transaction, Value: at*1000 - tx.CompleteNS, Bytes: tx.Bytes, Reason: "ns_to_us_rounding"})
			(&peerTransferEvent{at: at, fabric: n.fabric, job: j}).Execute(nil)
		})
	}})
	if err != nil {
		panic(err)
	}
	// Queue-aware policy estimates use an isolated estimate; actual completion
	// comes exclusively from the fine event pump below.
	d := n.estimate(direction)
	for _, resource := range j.path {
		n.fabric.busyUntil[resource] = now + d
	}
	n.arm(j.schedule)
}
func (n *NativePeerRuntime) estimate(direction string) int64 {
	c := n.points[direction].Cost
	ns := c.PlanNS + c.SetupNS + max(56*c.SubmitNS, c.SubmitNS+56*c.DMANS) + c.ObserveNS
	return (ns + 999) / 1000
}
func (n *NativePeerRuntime) arm(schedule func(sim.Event)) {
	next, ok := n.Runtime.Engine.NextNS()
	if !ok {
		return
	}
	at := (next + 999) / 1000
	if !n.scheduled[at] {
		n.scheduled[at] = true
		schedule(&nativePumpEvent{at: at, n: n, schedule: schedule})
	}
}

type nativePumpEvent struct {
	at       int64
	n        *NativePeerRuntime
	schedule func(sim.Event)
}

func (e *nativePumpEvent) Timestamp() int64 { return e.at }
func (e *nativePumpEvent) Priority() int    { return sim.PriorityAdapterLoad }
func (e *nativePumpEvent) Execute(_ *sim.Simulator) {
	n := e.n
	delete(n.scheduled, e.at)
	n.advance(e.at)
	completed := n.completed
	n.completed = nil
	for _, done := range completed {
		done()
	}
	n.arm(e.schedule)
}
func (f *PeerFabric) NativeCheck() error {
	if f.native == nil {
		return nil
	}
	n := f.native
	m := n.Runtime.Memory
	if err := m.Check(); err != nil {
		return err
	}
	if n.Runtime.Engine.Pending() != 0 || len(n.completed) != 0 || len(n.scheduled) != 0 || n.forwardActive {
		return fmt.Errorf("native event pipeline not drained")
	}
	for _, b := range m.Buffers {
		if b.Pins != 0 || b.Writers != 0 || b.Holds != 0 {
			return fmt.Errorf("native buffer references remain")
		}
	}
	if len(n.hostLeased) != len(f.pools["dram"].entries) {
		return fmt.Errorf("L2 physical leases disagree with directory")
	}
	for id, hash := range f.stores[0].ready {
		if n.pagedEnabled {
			if err := n.checkPagedContents(hash, n.hbmID(id)); err != nil {
				return err
			}
			continue
		}
		origin := n.origins[hash]
		if origin == 0 {
			return fmt.Errorf("unknown HBM payload identity")
		}
		b := m.Buffers[n.hbmID(id)]
		for i, c := range b.Cells {
			if c != (kvruntime.Cell{Origin: origin, Offset: int64(i) * 256, Valid: true}) {
				return fmt.Errorf("HBM published payload mismatch slot=%d", id)
			}
		}
	}
	for hash, entry := range f.pools["dram"].entries {
		if !entry.ready {
			return fmt.Errorf("unpublished entry %s", hash)
		}
		if n.pagedEnabled {
			if err := n.checkPagedContents(hash, entry.nativeBuffer); err != nil {
				return err
			}
			continue
		}
		origin := n.origins[hash]
		if origin == 0 {
			return fmt.Errorf("unknown L2 payload identity")
		}
		b := m.Buffers[entry.nativeBuffer]
		for i, c := range b.Cells {
			if c != (kvruntime.Cell{Origin: origin, Offset: int64(i) * 256, Valid: true}) {
				return fmt.Errorf("L2 payload mismatch %s", hash)
			}
		}
	}
	return nil
}
