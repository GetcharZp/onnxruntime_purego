<div align="center" style="text-align: center;">
  <img src="./assets/logo.png" alt="logo" width="200" style="display: block; margin: 0 auto;" />
</div>

<p align="center">
   <a href="https://github.com/getcharzp/onnxruntime_purego/fork" target="blank">
      <img src="https://img.shields.io/github/forks/getcharzp/onnxruntime_purego?style=for-the-badge" alt="onnxruntime_purego forks"/>
   </a>
   <a href="https://github.com/getcharzp/onnxruntime_purego/stargazers" target="blank">
      <img src="https://img.shields.io/github/stars/getcharzp/onnxruntime_purego?style=for-the-badge" alt="onnxruntime_purego stars"/>
   </a>
   <a href="https://github.com/getcharzp/onnxruntime_purego/pulls" target="blank">
      <img src="https://img.shields.io/github/issues-pr/getcharzp/onnxruntime_purego?style=for-the-badge" alt="onnxruntime_purego pull-requests"/>
   </a>
</p>

<p align="center">
  English | <a href="./docs/README_zh.md">简体中文</a>
</p>

A CGO-free, pure Go project built on top of `purego`. It binds directly to and calls the native onnxruntime
library interfaces through `purego`, so ONNX models can be loaded and inferred without a CGO build environment.
It is implemented based on the `onnxruntime` 1.28.3 headers.

## Installation

Download the [onnxruntime 1.28](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.3) dynamic library and install the `onnxruntime_purego` package for onnxruntime 1.28:

| onnxruntime | onnxruntime_purego |
|-------------|--------------------|
| 1.28        | v1.28.0            |

For older releases, see the [version mapping table](./docs/version-mapping.md).

```shell
# Download the latest version
go get -u github.com/getcharzp/onnxruntime_purego

# Download the specific release for onnxruntime 1.28
go get -u github.com/getcharzp/onnxruntime_purego@v1.28.0
```

## Quick Start

```go
package main

import (
	ort "github.com/getcharzp/onnxruntime_purego"
	"log"
)

const testModelPath = "./testdata/yolo11n.onnx"

func main() {
	engine, _ := ort.NewEngine(ort.DefaultLibraryPath())
	defer engine.Destroy()
	session, err := engine.NewSession(testModelPath, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Destroy()

	// Prepare the data from the model's own declaration
	input := session.Inputs[0]
	count, ok := input.ElementCount()
	if !ok {
		log.Fatalf("input %q has a dynamic shape %v", input.Name, input.Shape)
	}

	inputValue, err := ort.NewTensor(input.Shape, make([]float32, count))
	if err != nil {
		log.Fatal(err)
	}
	defer inputValue.Destroy()

	outputs, err := session.Run(map[string]*ort.Value{input.Name: inputValue})
	if err != nil {
		log.Fatal(err)
	}

	for name, output := range outputs {
		defer output.Destroy()

		outputData, err := ort.GetTensorData[float32](output)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("%v: %+v", name, outputData[:min(len(outputData), 20)])
	}
}
```

## Usage

### Loading a Model from Memory

Use `NewSessionFromBytes` to load a model embedded with `go:embed`, avoiding file path dependencies:

```go
import _ "embed"

//go:embed models/yolo11n.onnx
var modelData []byte

session, err := engine.NewSessionFromBytes(modelData, nil)
```

The returned session holds a reference to `modelData`, so the buffer stays valid for its lifetime; do not modify it while the session is alive.

### Dynamic Shapes

Models exported with dynamic dimensions report them as `-1`, which makes `TensorInfo.ElementCount` fail. Fix the dimension when creating the session:

```go
opts, _ := engine.NewSessionOptions()
defer opts.Destroy()
opts.AddFreeDimensionOverrideByName("batch", 1)

session, _ := engine.NewSession("model.onnx", opts)
// the dimension is now fixed to 1 instead of -1 in session.Inputs[i].Shape
```

Use `AddFreeDimensionOverride` instead when the dimension carries a denotation such as `DATA_BATCH`.

### Execution Providers

Check which execution providers the local onnxruntime build supports before enabling one:

```go
opts, _ := engine.NewSessionOptions()
defer opts.Destroy()

for _, p := range engine.GetAvailableProviders() {
    if p == "CUDAExecutionProvider" {
        opts.EnableCUDA(nil)
    }
}
```

## Examples

### YOLOv11 Object Detection

| Original Image                                      | Detection Result                                            |
|-----------------------------------------------------|------------------------------------------------------------|
| <img width="100%" src="./testdata/test.png" alt=""> | <img width="100%" src="./testdata/yolov11_det.png" alt=""> |

