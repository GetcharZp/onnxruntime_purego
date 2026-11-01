//go:build linux || darwin

package ort

import "unsafe"

// stringToPathPtr 将路径转换为 UTF-8 (char*) 指针
func stringToPathPtr(s string) (unsafe.Pointer, error) {
	return unsafe.Pointer(cString(s)), nil
}
