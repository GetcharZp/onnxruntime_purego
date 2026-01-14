package ort

import (
	"fmt"
	"runtime"
	"unsafe"
)

// DefaultLibraryPath 默认动态库路径, 例如：
//
// # Return:
//
//	Windows: ./lib/onnxruntime.dll
//	Linux amd64: ./lib/onnxruntime_amd64.so
//	Linux arm64: ./lib/onnxruntime_arm64.so
//	Mac amd64: ./lib/onnxruntime_amd64.dylib
//	Mac arm64: ./lib/onnxruntime_arm64.dylib
func DefaultLibraryPath() string {
	baseDir := "./lib/"
	libName := "onnxruntime"

	// windows onnxruntime.dll
	if runtime.GOOS == "windows" {
		return baseDir + libName + ".dll"
	}

	// linux darwin ext
	var ext string
	switch runtime.GOOS {
	case "darwin":
		ext = "dylib"
	case "linux":
		ext = "so"
	default:
		return baseDir + libName + "_amd64.so" // 默认返回 linux amd64
	}

	// 拼接完整路径: ./lib/onnxruntime + _ + amd64/arm64 + . + so/dylib
	return fmt.Sprintf("%s%s_%s.%s", baseDir, libName, runtime.GOARCH, ext)
}

// stringToCString 将字符串转换为字节指针
func stringToCString(s string) (*byte, error) {
	b := make([]byte, len(s)+1)
	copy(b, s)
	b[len(s)] = 0
	return &b[0], nil
}

// cStringToString 将字节指针转换为字符串
func cStringToString(ptr *byte) string {
	if ptr == nil {
		return ""
	}
	var length int
	for {
		if *(*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(length))) == 0 {
			break
		}
		length++
	}
	return string(unsafe.Slice(ptr, length))
}
