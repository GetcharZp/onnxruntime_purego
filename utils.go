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

// slicePtr 返回切片首元素的地址
//
// 空切片返回 nil —— 直接写 &s[0] 会 panic，而 C API 允许把 nil 与长度 0 一起传入。
func slicePtr[T any](s []T) *T {
	if len(s) == 0 {
		return nil
	}
	return &s[0]
}

// cString 将 Go 字符串转换为以 '\0' 结尾的 C 字符串
func cString(s string) *byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0]
}

// goString 将 C 字符串转换为 Go 字符串，nil 指针返回空串
func goString(ptr *byte) string {
	if ptr == nil {
		return ""
	}
	length := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(ptr), length)) != 0 {
		length++
	}
	return string(unsafe.Slice(ptr, length))
}
