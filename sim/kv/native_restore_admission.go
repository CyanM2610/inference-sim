package kv

import "fmt"

const NativeRestoreAdmissionContract = "native_full_restore_v1"

// EnableNativeRestoreAdmission makes an asynchronous lookup one atomic HBM
// admission, as in vLLM's OffloadingConnector. Transfer windowing must not split
// that admission into non-preemptible partial owners with competing promises.
func (s *PeerCache) EnableNativeRestoreAdmission() error {
	p := s.fabric.phases
	if !s.restoreControl || p == nil || p.started || p.active || s.fabric.Pending() != 0 {
		return fmt.Errorf("native restore admission requires unused restore-control engine phases")
	}
	s.nativeRestoreAdmission = true
	return nil
}
