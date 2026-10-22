package kvruntime

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// Cell is a lossless symbolic byte-range identity: Origin selects a payload,
// Offset its first byte. Copies preserve the identity of every byte in a cell.
// The simulator does not manufacture floating-point attention/KV values.
type Cell struct {
	Origin uint64
	Offset int64
	Valid  bool
}
type Buffer struct {
	CellBytes           int64
	ID, Pool, Kind      string
	Bytes               int64
	State               string
	Touched, Registered bool
	Pins, Writers       int
	Holds               int
	WriterOwner         string
	Cells               []Cell
	Opaque              bool
}
type Memory struct {
	Unit                   int64
	Capacity, Used         map[string]int64
	Peak                   map[string]int64
	Retained, PhysicalPeak map[string]int64
	FixedAllocator         map[string]bool
	Buffers                map[string]*Buffer
	e                      *Engine
}

func NewMemory(e *Engine, unit int64, capacity map[string]int64) (*Memory, error) {
	if unit <= 0 {
		return nil, fmt.Errorf("positive content granule required")
	}
	c := map[string]int64{}
	for k, v := range capacity {
		if k == "" || v <= 0 {
			return nil, fmt.Errorf("invalid capacity")
		}
		c[k] = v
	}
	return &Memory{Unit: unit, Capacity: c, Used: map[string]int64{}, Peak: map[string]int64{}, Retained: map[string]int64{}, PhysicalPeak: map[string]int64{}, FixedAllocator: map[string]bool{}, Buffers: map[string]*Buffer{}, e: e}, nil
}
func (m *Memory) Reserve(id, pool, kind string, bytes int64) (*Buffer, error) {
	return m.reserve(id, pool, kind, bytes, m.Unit, false)
}

// ReserveGranular keeps tiny control payloads (such as an int64 output token)
// in the same physical/reference ledger without inflating them to a KV granule.
// Copy endpoints must use the same granularity; implicit reinterpretation is
// rejected because it could lose byte-range identity.
func (m *Memory) ReserveGranular(id, pool, kind string, bytes, cellBytes int64) (*Buffer, error) {
	return m.reserve(id, pool, kind, bytes, cellBytes, false)
}

// Declare describes a future buffer endpoint while constructing an async DAG.
// It consumes no physical capacity. Allocate commits the reservation at the
// producer's actual execution time; use, publication and holds remain illegal
// before that transition. This is not an overcommit permission.
func (m *Memory) Declare(id, pool, kind string, bytes, cellBytes int64) (*Buffer, error) {
	if id == "" || m.Buffers[id] != nil || bytes <= 0 || cellBytes <= 0 || bytes%cellBytes != 0 || m.Capacity[pool] < bytes {
		return nil, fmt.Errorf("invalid future buffer %s", id)
	}
	b := &Buffer{ID: id, Pool: pool, Kind: kind, Bytes: bytes, CellBytes: cellBytes, State: "declared"}
	m.Buffers[id] = b
	m.e.Emit(Record{Name: "memory_declare", Destination: id, Resource: pool, Bytes: bytes, CellBytes: cellBytes, Reason: kind})
	return b, nil
}

// ReserveOpaque accounts for measured non-KV resident allocations without
// inventing their tensor contents. Opaque storage cannot enter a KV transfer.
func (m *Memory) ReserveOpaque(id, pool, kind string, bytes int64) (*Buffer, error) {
	return m.reserve(id, pool, kind, bytes, m.Unit, true)
}
func (m *Memory) reserve(id, pool, kind string, bytes, cellBytes int64, opaque bool) (*Buffer, error) {
	if id == "" || m.Buffers[id] != nil || bytes <= 0 || cellBytes <= 0 || bytes%cellBytes != 0 {
		return nil, fmt.Errorf("invalid/duplicate buffer or unaligned size %s", id)
	}
	b := &Buffer{ID: id, Pool: pool, Kind: kind, Bytes: bytes, State: "reserved", Opaque: opaque, CellBytes: cellBytes}
	m.Buffers[id] = b
	if err := m.commitReservation(b); err != nil {
		delete(m.Buffers, id)
		return nil, err
	}
	return b, nil
}
func (m *Memory) commitReservation(b *Buffer) error {
	pool, bytes := b.Pool, b.Bytes
	if m.Capacity[pool]-m.Used[pool] < bytes {
		return fmt.Errorf("capacity exhausted in %s", pool)
	}
	if m.FixedAllocator[pool] && m.Used[pool]+bytes > m.Retained[pool] {
		return fmt.Errorf("unprofiled allocator growth in %s", pool)
	}
	if !b.Opaque {
		b.Cells = make([]Cell, bytes/b.CellBytes)
	}
	b.State = "reserved"
	m.Used[pool] += bytes
	m.Peak[pool] = max(m.Peak[pool], m.Used[pool])
	m.PhysicalPeak[pool] = max(m.PhysicalPeak[pool], m.PhysicalUsed(pool))
	granule := int64(0)
	if b.CellBytes != m.Unit {
		granule = b.CellBytes
	}
	m.e.Emit(Record{Name: "memory_reserve", Destination: b.ID, Resource: pool, Bytes: bytes, Reason: b.Kind, CellBytes: granule})
	return nil
}
func (m *Memory) Allocate(id string) error {
	b := m.Buffers[id]
	if b != nil && b.State == "declared" {
		if err := m.commitReservation(b); err != nil {
			return err
		}
		b.State = "reserved"
	}
	if b == nil || b.State != "reserved" {
		return fmt.Errorf("allocation without reservation: %s", id)
	}
	b.State = "allocated"
	m.e.Emit(Record{Name: "memory_allocate", Destination: id, Resource: b.Pool, Bytes: b.Bytes})
	return nil
}
func (m *Memory) Fill(id string, origin uint64) error {
	b := m.Buffers[id]
	if b == nil || b.State != "allocated" || b.Pins > 0 || b.Writers > 0 || b.Holds > 0 {
		return fmt.Errorf("fill of unavailable buffer %s", id)
	}
	for i := range b.Cells {
		b.Cells[i] = Cell{origin, int64(i) * b.CellBytes, true}
	}
	b.Touched = true
	b.State = "ready"
	m.e.Emit(Record{Name: "memory_publish", Destination: id, Bytes: b.Bytes, Reason: "initialized", Origin: origin})
	return nil
}
func (m *Memory) Register(id string) error {
	b := m.Buffers[id]
	if b == nil || b.State == "reserved" || b.State == "declared" || b.Pool != "dram" || b.Registered || !b.Touched {
		return fmt.Errorf("invalid registration %s", id)
	}
	b.Registered = true
	m.e.Emit(Record{Name: "memory_register", Destination: id, Bytes: b.Bytes})
	return nil
}
func (m *Memory) Unregister(id string) error {
	b := m.Buffers[id]
	if b == nil || !b.Registered || b.Pins > 0 || b.Writers > 0 || b.Holds > 0 {
		return fmt.Errorf("invalid unregister %s", id)
	}
	b.Registered = false
	m.e.Emit(Record{Name: "memory_unregister", Destination: id, Bytes: b.Bytes})
	return nil
}
func (m *Memory) Pin(id string) error {
	b := m.Buffers[id]
	if b == nil || b.State != "ready" || b.Writers > 0 {
		return fmt.Errorf("read before publication: %s", id)
	}
	b.Pins++
	m.e.Emit(Record{Name: "memory_pin", Source: id, Bytes: b.Bytes, ContentSHA256: m.Digest(id)})
	return nil
}
func (m *Memory) Unpin(id string) error {
	b := m.Buffers[id]
	if b == nil || b.Pins <= 0 {
		return fmt.Errorf("unbalanced pin: %s", id)
	}
	b.Pins--
	m.e.Emit(Record{Name: "memory_unpin", Source: id, Bytes: b.Bytes})
	return nil
}
func (m *Memory) BeginWrite(id string) error {
	return m.BeginWriteOwned(id, "")
}

// BeginWriteOwned allows one compound operation to own several disjoint
// kernel writes and publish their destination only after the whole DAG ends.
func (m *Memory) BeginWriteOwned(id, owner string) error {
	b := m.Buffers[id]
	if b == nil || b.State == "reserved" || b.State == "declared" || b.Pins > 0 || b.Writers > 0 || b.Holds > 0 {
		return fmt.Errorf("target unavailable: %s", id)
	}
	b.Writers++
	b.WriterOwner = owner
	b.State = "writing"
	m.e.Emit(Record{Name: "memory_write_pending", Destination: id, Bytes: b.Bytes})
	return nil
}
func (m *Memory) Publish(id string) error {
	return m.PublishOwned(id, "")
}
func (m *Memory) PublishOwned(id, owner string) error {
	b := m.Buffers[id]
	if b == nil || b.Writers != 1 || b.WriterOwner != owner {
		return fmt.Errorf("publication without writer: %s", id)
	}
	b.Writers = 0
	b.WriterOwner = ""
	b.State = "ready"
	b.Touched = true
	m.e.Emit(Record{Name: "memory_publish", Destination: id, Bytes: b.Bytes, Reason: "completion_adopted", ContentSHA256: m.Digest(id)})
	return nil
}
func (m *Memory) Copy(src, dst string, so, do, n int64) error {
	return m.copy(src, dst, so, do, n, false)
}

// CopyPadding preserves undefined cells when a physical gather copies an
// entire last page. Consumers must still use strict Copy for meaningful KV.
func (m *Memory) CopyPadding(src, dst string, so, do, n int64) error {
	return m.copy(src, dst, so, do, n, true)
}
func (m *Memory) copy(src, dst string, so, do, n int64, padding bool) error {
	a, b := m.Buffers[src], m.Buffers[dst]
	if a == nil || b == nil || a.Opaque || b.Opaque || a.Pins <= 0 || a.State != "ready" || b.Writers != 1 {
		return fmt.Errorf("copy without live pin/writer")
	}
	if a.CellBytes != b.CellBytes || n <= 0 || so < 0 || do < 0 || so%a.CellBytes != 0 || do%b.CellBytes != 0 || n%a.CellBytes != 0 || so > a.Bytes-n || do > b.Bytes-n {
		return fmt.Errorf("copy range out of bounds/unaligned")
	}
	for _, c := range a.Cells[so/a.CellBytes : (so+n)/a.CellBytes] {
		if !c.Valid && !padding {
			return fmt.Errorf("copy of uninitialized source")
		}
	}
	copy(b.Cells[do/b.CellBytes:(do+n)/b.CellBytes], a.Cells[so/a.CellBytes:(so+n)/a.CellBytes])
	m.e.Emit(Record{Name: "memory_copy", Source: src, Destination: dst, SourceOffset: so, DestinationOffset: do, Bytes: n, AllowUndefined: padding})
	return nil
}

// ComputeRange records newly computed KV identity under the same exclusive
// writer contract as DMA. Publication remains a separate dependent operation.
func (m *Memory) ComputeRange(id string, start, bytes int64, origin uint64, offset int64) error {
	b := m.Buffers[id]
	if b == nil || b.Opaque || b.Writers != 1 || start < 0 || bytes <= 0 || start > b.Bytes-bytes || start%b.CellBytes != 0 || bytes%b.CellBytes != 0 || offset < 0 {
		return fmt.Errorf("invalid computed range")
	}
	for i := int64(0); i < bytes; i += b.CellBytes {
		b.Cells[(start+i)/b.CellBytes] = Cell{origin, offset + i, true}
	}
	m.e.Emit(Record{Name: "memory_compute_range", Destination: id, DestinationOffset: start, Bytes: bytes, Origin: origin, OriginOffset: offset})
	return nil
}
func (m *Memory) Free(id string) error {
	b := m.Buffers[id]
	if b == nil || b.Pins > 0 || b.Writers > 0 || b.Holds > 0 || b.Registered {
		return fmt.Errorf("free with live references/registration: %s", id)
	}
	if b.State == "declared" {
		delete(m.Buffers, id)
		m.e.Emit(Record{Name: "memory_undeclare", Source: id, Resource: b.Pool, Bytes: b.Bytes})
		return nil
	}
	m.Used[b.Pool] -= b.Bytes
	delete(m.Buffers, id)
	m.e.Emit(Record{Name: "memory_free", Source: id, Resource: b.Pool, Bytes: b.Bytes})
	return nil
}
func (m *Memory) Check() error {
	used := map[string]int64{}
	for _, b := range m.Buffers {
		if b.State == "declared" {
			if b.Pins != 0 || b.Holds != 0 || b.Writers != 0 || len(b.Cells) != 0 {
				return fmt.Errorf("future buffer used before allocation")
			}
			continue
		}
		if b.CellBytes <= 0 || b.Bytes%b.CellBytes != 0 || (!b.Opaque && int64(len(b.Cells)) != b.Bytes/b.CellBytes) {
			return fmt.Errorf("invalid content granularity %s", b.ID)
		}
		if b.Pins < 0 || b.Writers < 0 || b.Holds < 0 {
			return fmt.Errorf("negative references")
		}
		used[b.Pool] += b.Bytes
	}
	for p, n := range m.Used {
		if n < 0 || n > m.Capacity[p] || used[p] != n {
			return fmt.Errorf("physical accounting mismatch %s", p)
		}
	}
	for p, n := range m.Retained {
		if n < 0 || n > m.Capacity[p] || (m.FixedAllocator[p] && m.Used[p] > n) {
			return fmt.Errorf("allocator reservation mismatch %s", p)
		}
	}
	return nil
}

// RetainAllocator models a caching allocator's reserved backing separately
// from its live tensor leases. Physical occupancy is max(reserved, active),
// never their sum. Fixed warm pools reject uncalibrated growth.
func (m *Memory) RetainAllocator(pool string, bytes int64, fixed bool) error {
	if pool == "" || m.Capacity[pool] <= 0 || bytes < 0 || bytes > m.Capacity[pool] || (fixed && bytes < m.Used[pool]) {
		return fmt.Errorf("invalid allocator reservation")
	}
	m.Retained[pool] = bytes
	m.FixedAllocator[pool] = fixed
	m.PhysicalPeak[pool] = max(m.PhysicalPeak[pool], m.PhysicalUsed(pool))
	m.e.Emit(Record{Name: "memory_allocator_reserve", Resource: pool, Bytes: bytes, Reason: "cached_backing; overlaps_live_allocations_not_additive"})
	return nil
}
func (m *Memory) PhysicalUsed(pool string) int64 { return max(m.Used[pool], m.Retained[pool]) }

func (m *Memory) SeedRange(id string, start, bytes int64, origin uint64, offset int64) error {
	b := m.Buffers[id]
	if b == nil || b.State != "ready" || b.Pins > 0 || b.Writers > 0 || b.Holds > 0 || start < 0 || bytes <= 0 || start > b.Bytes-bytes || start%b.CellBytes != 0 || bytes%b.CellBytes != 0 || offset < 0 {
		return fmt.Errorf("invalid input payload range")
	}
	for i := int64(0); i < bytes; i += b.CellBytes {
		b.Cells[(start+i)/b.CellBytes] = Cell{origin, offset + i, true}
	}
	m.e.Emit(Record{Name: "memory_initialize_range", Destination: id, DestinationOffset: start, Bytes: bytes, Origin: origin, OriginOffset: offset})
	return nil
}
func (m *Memory) Digest(id string) string {
	b := m.Buffers[id]
	h := sha256.New()
	var bytes [17]byte
	for _, c := range b.Cells {
		binary.LittleEndian.PutUint64(bytes[:8], c.Origin)
		binary.LittleEndian.PutUint64(bytes[8:16], uint64(c.Offset))
		bytes[16] = 0
		if c.Valid {
			bytes[16] = 1
		}
		_, _ = h.Write(bytes[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// Hold protects storage for a queued consumer without asserting that its
// producer has published data. Pin still checks readiness at execution time.
func (m *Memory) Hold(id string) error {
	b := m.Buffers[id]
	if b == nil || b.State == "reserved" || b.State == "declared" {
		return fmt.Errorf("hold of unavailable storage")
	}
	b.Holds++
	m.e.Emit(Record{Name: "memory_hold", Source: id, Bytes: b.Bytes})
	return nil
}

// Invalidate changes logical ownership of a physical arena slot; it does not
// zero GPU bytes or allocate/free storage. Stale payload must not become a hit.
func (m *Memory) Invalidate(id string) error {
	b := m.Buffers[id]
	if b == nil || b.State == "declared" || b.Opaque || b.Pins > 0 || b.Writers > 0 || b.Holds > 0 {
		return fmt.Errorf("invalidate referenced storage %s", id)
	}
	clear(b.Cells)
	b.State = "allocated"
	m.e.Emit(Record{Name: "memory_invalidate", Destination: id, Bytes: b.Bytes, Reason: "physical_slot_reused; old_contents_undefined"})
	return nil
}
func (m *Memory) Unhold(id string) error {
	b := m.Buffers[id]
	if b == nil || b.Holds <= 0 {
		return fmt.Errorf("unbalanced storage hold")
	}
	b.Holds--
	m.e.Emit(Record{Name: "memory_unhold", Source: id, Bytes: b.Bytes})
	return nil
}
