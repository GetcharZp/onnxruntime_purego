//go:build linux || darwin

package sys

import (
	"fmt"

	"github.com/ebitengine/purego"
)

// LoadLibrary 加载动态库文件
func LoadLibrary(name string) (uintptr, error) {
	handle, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return 0, fmt.Errorf("failed to load shared library %s: %w", name, err)
	}
	return handle, nil
}
