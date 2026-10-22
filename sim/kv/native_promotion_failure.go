package kv

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// ConfigureNativePromotionPartialFailure arms one execution-environment fault
// for the next captured native control, before any controls have been queued.
// It submits a proper nonempty prefix of host objects, then drains a failed
// operation. A candidate with too few retained objects is rejected without HBM
// mutation; the fault remains armed. Policies do not choose terminal outcomes.
func (s *PeerCache) ConfigureNativePromotionPartialFailure(submittedObjects int) error {
	if !s.nativePhaseTransfers || s.nativeExecutor == nil || !s.nativeExecutor.HasTransferFailureModel() ||
		s.nativeControlSerial != 0 || s.nativePartialFailure != 0 || submittedObjects <= 0 {
		return fmt.Errorf("partial Promotion failure requires unused native control, explicit failure costs and positive submitted object count")
	}
	s.nativePartialFailure = submittedObjects
	return nil
}

// Ownership remains the original whole operation, but the executor/accounting
// sees only the submitted prefix. All member callbacks still settle together.
func (s *PeerCache) prepareNativePartialFailure(j *peerJob) *sim.NativeTransferFailure {
	if s.nativePartialFailure == 0 || j.nativeControl == nil {
		return nil
	}
	objects := j.record.NativeObjects
	n := s.nativePartialFailure
	if n >= len(objects) || j.record.Destination != "hbm" || j.nativeControl.copiedTargets != nil {
		panic("partial failure lost its preflighted whole operation")
	}
	failure := &sim.NativeTransferFailure{ReservedShape: sim.NativeTransferShape{Direction: "load", Objects: objects}, SubmittedObjects: n}
	s.nativePartialFailure = 0
	j.nativeControl.copiedTargets = map[int64]bool{}
	j.record.NativeFailure = failure
	j.record.NativeObjects = append([]sim.NativeTransferObject(nil), objects[:n]...)
	j.record.Bytes, j.record.EffectiveBytes = 0, 0
	j.record.HBMBlocks, j.record.HostObjects = nil, nil
	for _, object := range objects[:n] {
		j.record.Bytes += object.StagingBytes
		j.record.EffectiveBytes += object.KernelBytes
		j.record.HBMBlocks = append(j.record.HBMBlocks, object.HBMBlocks...)
		j.record.HostObjects = append(j.record.HostObjects, object.Key)
		for _, id := range object.HBMBlocks {
			j.nativeControl.copiedTargets[id] = true
		}
	}
	sort.Strings(j.record.HostObjects)
	return failure
}
