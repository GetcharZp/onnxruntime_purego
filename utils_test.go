package ort

import (
	"math"
	"testing"

	"github.com/up-zero/gotool/testutil"
)

func TestSlicePtr(t *testing.T) {
	s := []int{1, 2, 3}
	testutil.Equal(t, *slicePtr(s), 1)

	// 空切片必须返回 nil：直接取 &s[0] 会 panic，而 C API 允许 nil + 长度 0
	testutil.Equal(t, slicePtr([]int{}), (*int)(nil))
	testutil.Equal(t, slicePtr[int](nil), (*int)(nil))
}

func TestCString(t *testing.T) {
	testutil.Equal(t, goString(cString("hello")), "hello")

	// 空字符串
	testutil.Equal(t, goString(cString("")), "")

	// 非 ASCII
	testutil.Equal(t, goString(cString("你好")), "你好")
}

func TestGoString(t *testing.T) {
	// nil 指针返回空字符串，而不是 panic
	testutil.Equal(t, goString(nil), "")

	// 遇到 '\0' 即终止
	b := []byte{'g', 'o', 0, 'x'}
	testutil.Equal(t, goString(&b[0]), "go")

	// 返回值必须是拷贝：源缓冲区之后会被 onnxruntime 释放或复用
	s := goString(&b[0])
	b[0] = 'G'
	testutil.Equal(t, s, "go")
}

func TestShapeElementCount(t *testing.T) {
	cases := []struct {
		name  string
		shape []int64
		want  int64
		ok    bool
	}{
		{"4d", []int64{1, 3, 640, 640}, 1228800, true},
		{"scalar", []int64{}, 1, true},
		{"zero dim", []int64{0}, 0, true},
		{"dynamic dim", []int64{1, -1, 640, 640}, 0, false},
		{"overflow", []int64{math.MaxInt64, 2}, 0, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := shapeElementCount(tc.shape)
			testutil.Equal(t, ok, tc.ok)
			testutil.Equal(t, got, tc.want)
		})
	}
}
