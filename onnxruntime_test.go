package ort

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/up-zero/gotool/testutil"
)

// newTestEngine 创建测试用 Engine，并在测试结束后自动释放。
func newTestEngine(t *testing.T) *Engine {
	t.Helper()

	engine, err := NewEngine(DefaultLibraryPath())
	testutil.Equal(t, err, nil)
	t.Cleanup(engine.Destroy)

	return engine
}

func TestDefaultLibraryPath(t *testing.T) {
	path := DefaultLibraryPath()

	testutil.Equal(t, strings.HasPrefix(path, "./lib/onnxruntime"), true)

	var wantExt string
	switch runtime.GOOS {
	case "windows":
		wantExt = ".dll"
	case "darwin":
		wantExt = ".dylib"
	default:
		wantExt = ".so"
	}
	testutil.Equal(t, filepath.Ext(path), wantExt)

	// linux / darwin 的库名带架构后缀，例如 onnxruntime_amd64.so
	if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
		testutil.Equal(t, strings.Contains(path, runtime.GOARCH), true)
	}
}

func TestEngine_GetVersion(t *testing.T) {
	engine := newTestEngine(t)

	version := engine.GetVersion()
	testutil.NotEqual(t, version, "")
	testutil.NotEqual(t, version, "unknown")

	// 版本号形如 1.25.1，主版本需与绑定的 header 主版本一致
	major, _, ok := strings.Cut(version, ".")
	testutil.Equal(t, ok, true)
	testutil.Equal(t, major, "1")
}

func TestNewEngine_InvalidLibraryPath(t *testing.T) {
	engine, err := NewEngine("./lib/not_exists.dll")
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, engine, (*Engine)(nil))
}

func TestEngine_Destroy(t *testing.T) {
	engine, err := NewEngine(DefaultLibraryPath())
	testutil.Equal(t, err, nil)
	testutil.Equal(t, getDefaultEngine(), engine)

	engine.Destroy()
	// 避免 defaultEngine 悬垂：Destroy 后包级 NewTensor 必须报错，
	// 而不是把已释放的 MemoryInfoHandle 交给 onnxruntime
	testutil.Equal(t, getDefaultEngine(), (*Engine)(nil))

	_, err = NewTensor([]int64{1}, []float32{1})
	testutil.NotEqual(t, err, nil)

	// 重复释放不应 panic
	engine.Destroy()
}
