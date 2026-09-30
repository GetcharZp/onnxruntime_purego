package ort

import (
	"math"
	"testing"
	"unsafe"

	"github.com/up-zero/gotool/testutil"
)

func TestSlicePtr(t *testing.T) {
	s := []int{1, 2, 3}
	testutil.Equal(t, *slicePtr(s), 1)

	// 空切片必须返回 nil：直接取 &s[0] 会 panic，而 C API 允许 nil + 长度 0
	testutil.Equal(t, slicePtr([]int{}), (*int)(nil))
	testutil.Equal(t, slicePtr[int](nil), (*int)(nil))
}

func TestStringToCString(t *testing.T) {
	ptr, err := stringToCString("hello")
	testutil.Equal(t, err, nil)
	testutil.Equal(t, cStringToString(ptr), "hello")

	// 空字符串
	ptr, err = stringToCString("")
	testutil.Equal(t, err, nil)
	testutil.Equal(t, cStringToString(ptr), "")

	// 非 ASCII
	ptr, err = stringToCString("你好")
	testutil.Equal(t, err, nil)
	testutil.Equal(t, cStringToString(ptr), "你好")
}

func TestCStringToString(t *testing.T) {
	// nil 指针返回空字符串，而不是 panic
	testutil.Equal(t, cStringToString(nil), "")

	// 遇到 '\0' 即终止
	b := []byte{'g', 'o', 0, 'x'}
	testutil.Equal(t, cStringToString(&b[0]), "go")
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

func TestParseInputData(t *testing.T) {
	cases := []struct {
		name     string
		data     any
		dataType TensorElementDataType
		typeSize uintptr
		length   int
	}{
		{"float32", []float32{1, 2}, TensorElementDataTypeFloat, 4, 2},
		{"float64", []float64{1, 2}, TensorElementDataTypeDouble, 8, 2},
		{"int64", []int64{1, 2}, TensorElementDataTypeInt64, 8, 2},
		{"int32", []int32{1, 2}, TensorElementDataTypeInt32, 4, 2},
		{"int16", []int16{1, 2}, TensorElementDataTypeInt16, 2, 2},
		{"int8", []int8{1, 2}, TensorElementDataTypeInt8, 1, 2},
		{"uint64", []uint64{1, 2}, TensorElementDataTypeUint64, 8, 2},
		{"uint32", []uint32{1, 2}, TensorElementDataTypeUint32, 4, 2},
		{"uint16", []uint16{1, 2}, TensorElementDataTypeUint16, 2, 2},
		{"uint8", []uint8{1, 2}, TensorElementDataTypeUint8, 1, 2},
		{"bool", []bool{true, false}, TensorElementDataTypeBool, 1, 2},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dataType, typeSize, length, dataPtr, err := parseInputData(tc.data)
			testutil.Equal(t, err, nil)
			testutil.Equal(t, dataType, tc.dataType)
			testutil.Equal(t, typeSize, tc.typeSize)
			testutil.Equal(t, length, tc.length)
			testutil.NotEqual(t, dataPtr, nil)
		})
	}

	t.Run("unsupported", func(t *testing.T) {
		dataType, typeSize, length, dataPtr, err := parseInputData("abc")
		testutil.NotEqual(t, err, nil)
		testutil.Equal(t, dataType, TensorElementDataTypeUndefined)
		testutil.Equal(t, typeSize, uintptr(0))
		testutil.Equal(t, length, 0)
		testutil.Equal(t, dataPtr, unsafe.Pointer(nil))
	})
}
