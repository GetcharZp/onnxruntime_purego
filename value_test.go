package ort

import (
	"testing"
)

func TestValue_GetShape(t *testing.T) {
	engine, _ := NewEngine(DefaultLibraryPath())
	defer engine.Destroy()
	v, _ := NewTensor([]int64{1, 1, 6}, []float32{1, 2, 3, 4, 5, 6})
	defer v.Destroy()

	shape, err := v.GetShape()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("shape: %+v", shape)
}

func TestValue_GetElementCount(t *testing.T) {
	engine, _ := NewEngine(DefaultLibraryPath())
	defer engine.Destroy()
	v, _ := NewTensor([]int64{1, 1, 6}, []float32{1, 2, 3, 4, 5, 6})
	defer v.Destroy()

	shape, err := v.GetElementCount()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("count: %+v", shape)
}

func TestGetTensorData(t *testing.T) {
	engine, _ := NewEngine(DefaultLibraryPath())
	defer engine.Destroy()
	v, _ := NewTensor([]int64{1, 1, 6}, []float32{1, 2, 3, 4, 5, 6})
	defer v.Destroy()

	data, err := GetTensorData[float32](v)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("data: %+v", data)
}
