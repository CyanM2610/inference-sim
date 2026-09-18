package policylab

import "syscall"

func overheadProcessCPUNS() (int64, error) {
	h, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0, err
	}
	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(h, &creation, &exit, &kernel, &user); err != nil {
		return 0, err
	}
	ticks := func(f syscall.Filetime) int64 { return int64(uint64(f.HighDateTime)<<32 | uint64(f.LowDateTime)) }
	return 100 * (ticks(kernel) + ticks(user)), nil
}
