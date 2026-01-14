//go:build windows

package sys

import (
	"fmt"
	"syscall"
)

// LoadLibrary 加载动态库文件
func LoadLibrary(name string) (uintptr, error) {
	handle, err := syscall.LoadLibrary(name)
	if err != nil {
		return 0, fmt.Errorf("failed to load dll %s: %w", name, err)
	}
	return uintptr(handle), nil
}
