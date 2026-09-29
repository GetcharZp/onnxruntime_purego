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

// FreeLibrary 卸载动态库文件
//
// 仅在初始化失败、库中尚未建立任何全局状态时调用；正常使用流程不应卸载
// onnxruntime，其静态析构会带来崩溃风险。
func FreeLibrary(handle uintptr) {
	if handle != 0 {
		_ = purego.Dlclose(handle)
	}
}
