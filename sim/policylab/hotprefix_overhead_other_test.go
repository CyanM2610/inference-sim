//go:build !windows

package policylab

import "fmt"

func overheadProcessCPUNS() (int64, error) {
	return 0, fmt.Errorf("this serial profiling protocol requires Windows GetProcessTimes; no zero CPU fallback")
}
