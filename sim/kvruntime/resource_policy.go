package kvruntime

import (
	"fmt"
	"sort"
)

// ConfigureResource changes arbitration, independently of service duration.
// stream_drain prefers ready work from the current command group, yielding
// when that group has no ready command. It is an explicit model hypothesis,
// not a claim about every CUDA device/driver. Configure before submitting work.
func (e *Engine) ConfigureResource(name, policy string) error {
	if name == "" || (policy != "fifo" && policy != "stream_drain") {
		return fmt.Errorf("invalid resource policy")
	}
	if e.taskSerial != 0 {
		return fmt.Errorf("resource policy must precede graph construction")
	}
	r := e.resources[name]
	if r == nil {
		r = &resource{name: name}
		e.resources[name] = r
		e.resourceNames = append(e.resourceNames, name)
		sort.Strings(e.resourceNames)
	}
	r.policy = policy
	e.Emit(Record{Name: "resource_policy", Resource: name, Reason: policy})
	return nil
}
