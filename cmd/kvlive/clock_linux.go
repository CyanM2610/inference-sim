//go:build linux

package main

import "golang.org/x/sys/unix"

const clockSource = "CLOCK_MONOTONIC"

func clockNS() int64 {
	var t unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_MONOTONIC, &t); err != nil {
		panic(err)
	}
	return t.Nano()
}
