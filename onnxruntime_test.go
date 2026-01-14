package ort

import (
	"fmt"
	"testing"
)

func TestEngine_GetVersion(t *testing.T) {
	engine, err := NewEngine(DefaultLibraryPath())
	if err != nil {
		t.Fatalf("Failed to init engine: %v", err)
	}
	defer engine.Destroy()
	fmt.Printf("ONNX RUNTIME VERSION: %+v\n", engine.GetVersion())
}
