//go:build !linux

package main

import "time"

const clockSource = "process_relative_monotonic"

var clockOrigin = time.Now()

func clockNS() int64 { return time.Since(clockOrigin).Nanoseconds() }
