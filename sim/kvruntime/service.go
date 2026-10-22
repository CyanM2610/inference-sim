package kvruntime

import (
	"fmt"
	"math"
	"math/bits"
)

const FullServicePPM int64 = 1_000_000

type serviceProgress struct {
	remainingNS, fractionPPM, updatedAt, ratePPM int64
	revision                                     uint64
}

func (e *Engine) ServiceRate(class string) int64 {
	if rate, ok := e.serviceRates[class]; ok {
		return rate
	}
	return FullServicePPM
}

// SetServiceRate preserves work already done, including sub-nanosecond service
// fractions. Zero pauses a class until a future event resumes it. Rates above
// solo speed are rejected: this interface models contention, not extrapolated
// acceleration. Rates are supplied by a separately validated contention model.
func (e *Engine) SetServiceRate(class string, ratePPM int64) error {
	if class == "" || ratePPM < 0 || ratePPM > FullServicePPM {
		return fmt.Errorf("invalid service class/rate")
	}
	if e.ServiceRate(class) == ratePPM {
		return nil
	}
	if e.serviceRates == nil {
		e.serviceRates = map[string]int64{}
	}
	e.serviceRates[class] = ratePPM
	e.Emit(Record{Name: "service_class_rate", ServiceClass: class, ServiceRatePPM: ratePPM})
	// Only running tasks are touched. Historical task count must not make a
	// per-copy contention transition progressively more expensive.
	for _, name := range e.resourceNames {
		t := e.resources[name].active
		if t == nil || t.ServiceClass != class || t.service == nil {
			continue
		}
		e.advanceService(t)
		t.service.ratePPM = ratePPM
		if err := e.scheduleService(t); err != nil {
			e.Fail(err)
			return err
		}
	}
	return nil
}

func (e *Engine) advanceService(t *Task) {
	s := t.service
	elapsed := e.NowNS - s.updatedAt
	if elapsed == 0 {
		return
	}
	// 128-bit multiplication avoids overflowing nanoseconds * parts-per-million.
	hi, lo := bits.Mul64(uint64(elapsed), uint64(s.ratePPM))
	var carry uint64
	lo, carry = bits.Add64(lo, uint64(s.fractionPPM), 0)
	hi += carry
	whole, fraction := bits.Div64(hi, lo, uint64(FullServicePPM))
	if whole >= uint64(s.remainingNS) {
		s.remainingNS = 0
		s.fractionPPM = 0
	} else {
		s.remainingNS -= int64(whole)
		s.fractionPPM = int64(fraction)
	}
	s.updatedAt = e.NowNS
	e.Emit(Record{Name: "task_service_progress", Task: t.ID, Operation: t.Operation, Resource: t.Resource, ServiceClass: t.ServiceClass, DurationNS: elapsed, ServiceRatePPM: s.ratePPM, RemainingNS: s.remainingNS, FractionPPM: s.fractionPPM})
}

func (e *Engine) scheduleService(t *Task) error {
	s := t.service
	s.revision++
	revision := s.revision
	e.Emit(Record{Name: "task_service_rate", Task: t.ID, Operation: t.Operation, Resource: t.Resource, ServiceClass: t.ServiceClass, ServiceRatePPM: s.ratePPM, RemainingNS: s.remainingNS, FractionPPM: s.fractionPPM})
	if s.ratePPM == 0 && s.remainingNS > 0 {
		return nil
	}
	var duration uint64
	if s.remainingNS > 0 {
		hi, lo := bits.Mul64(uint64(s.remainingNS), uint64(FullServicePPM))
		var borrow uint64
		lo, borrow = bits.Sub64(lo, uint64(s.fractionPPM), 0)
		hi -= borrow
		if hi >= uint64(s.ratePPM) {
			return fmt.Errorf("service completion exceeds representable time")
		}
		q, r := bits.Div64(hi, lo, uint64(s.ratePPM))
		limit := uint64(math.MaxInt64 - e.NowNS)
		if q > limit || (q == limit && r > 0) {
			return fmt.Errorf("service completion timestamp overflow")
		}
		duration = q
		if r > 0 {
			duration++
		}
	}
	return e.atIf(e.NowNS+int64(duration), func() { e.completeTask(t) }, func() bool { return t.state == "running" && s.revision == revision })
}
