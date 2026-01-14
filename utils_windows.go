//go:build windows

package ort

import (
	"syscall"
	"unsafe"
)

// stringToPathPtr 将字符串转换为 UTF-16 (wchar_t*) 指针
func stringToPathPtr(s string) (unsafe.Pointer, error) {
	ptr, err := syscall.UTF16PtrFromString(s)
	if err != nil {
		return nil, err
	}
	return unsafe.Pointer(ptr), nil
}
