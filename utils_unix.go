//go:build linux || darwin

package ort

import "unsafe"

// stringToCString 将字符串转换为 UTF-8 (char*) 指针
func stringToPathPtr(s string) (unsafe.Pointer, error) {
	b := make([]byte, len(s)+1)
	copy(b, s)
	b[len(s)] = 0
	return unsafe.Pointer(&b[0]), nil
}
