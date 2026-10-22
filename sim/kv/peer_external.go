package kv

import "fmt"

// ExternalTransfer is a physical command issued by the real PeerFabric policy.
// The caller must retain both arenas and acknowledge only after the native copy
// completes. Estimated gate durations never synthesize an acknowledgement.
type ExternalTransfer struct {
	ID          int64  `json:"id"`
	IssuedUS    int64  `json:"issued_us"`
	Source      string `json:"source"`
	Destination string `json:"destination"`
	Hash        string `json:"hash"`
	Request     string `json:"request"`
	Reason      string `json:"reason"`
	Bytes       int64  `json:"bytes"`
}
type externalPeerRuntime struct {
	free      []string
	leased    map[string]bool
	pending   map[int64]*peerJob
	delivered map[int64]bool
	commands  []ExternalTransfer
}

func (f *PeerFabric) ConfigureExternalTransfers() error {
	if f.native != nil || f.external != nil || f.serial != 0 || len(f.stores) != 1 || len(f.pools) != 1 || f.pools["dram"] == nil || f.bytes != 917504 || f.stores[0].BlockSizeTokens != 16 {
		return fmt.Errorf("external transfers require one idle native-shaped cache and DRAM pool")
	}
	if f.mechanisms.DirectoryUS != 0 || f.mechanisms.CompletionUS != 0 || f.mechanisms.PolicyUS != 0 {
		return fmt.Errorf("live control executes actual CPU code; synthetic control costs must be zero")
	}
	x := &externalPeerRuntime{leased: map[string]bool{}, pending: map[int64]*peerJob{}, delivered: map[int64]bool{}}
	for i := int64(0); i < f.pools["dram"].config.CapacityBlocks; i++ {
		x.free = append(x.free, fmt.Sprintf("dram/slot/%d", i))
	}
	f.external = x
	return nil
}
func (f *PeerFabric) hbmBuffer(id int64) string { return fmt.Sprintf("hbm/slot/%d", id) }
func (x *externalPeerRuntime) reserve() string {
	if len(x.free) == 0 {
		panic("external physical host pool exhausted")
	}
	id := x.free[0]
	x.free = x.free[1:]
	if x.leased[id] {
		panic("external host slot double lease")
	}
	x.leased[id] = true
	return id
}
func (x *externalPeerRuntime) release(id string) {
	if !x.leased[id] {
		panic("unleased external host slot")
	}
	delete(x.leased, id)
	x.free = append(x.free, id)
}
func (x *externalPeerRuntime) submit(j *peerJob) {
	id := j.record.Transaction
	if x.pending[id] != nil || j.nativeSource == nil || j.nativeDestination == nil {
		panic("invalid external transfer")
	}
	x.pending[id] = j
	x.commands = append(x.commands, ExternalTransfer{ID: id, IssuedUS: j.record.Time, Source: j.nativeSource(), Destination: j.nativeDestination(), Hash: j.record.Hash, Request: j.record.Request, Reason: j.record.Reason, Bytes: j.record.Bytes})
}
func (f *PeerFabric) DrainExternalTransfers() []ExternalTransfer {
	if f.external == nil {
		return nil
	}
	x := f.external
	out := x.commands
	x.commands = nil
	for _, c := range out {
		x.delivered[c.ID] = true
	}
	return out
}

// CompleteExternalTransfer uses the caller's observation time. Device finish and
// observation/IPC latency are recorded separately by the hardware driver.
func (f *PeerFabric) CompleteExternalTransfer(id, at int64) error {
	if f.external == nil {
		return fmt.Errorf("external backend is disabled")
	}
	x := f.external
	j := x.pending[id]
	if j == nil || !x.delivered[id] || at <= j.record.Time {
		return fmt.Errorf("unknown, undelivered or premature external completion %d", id)
	}
	delete(x.pending, id)
	delete(x.delivered, id)
	j.schedule(&peerTransferEvent{at: at, fabric: f, job: j})
	return nil
}
