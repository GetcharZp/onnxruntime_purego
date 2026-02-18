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

基于 `purego` 实现的无 CGO 纯 Go 项目，通过 `purego` 直接绑定并调用 onnxruntime 原生库接口，无需依赖 CGO 编译环境，
即可实现 ONNX 模型的加载与推理计算，基于 `onnxruntime` 1.24.1 的头文件实现。

## 安装

下载 [onnxruntime 1.24](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1) 动态链接库，安装 `onnxruntime_purego` 库，版本对应关系：

| onnxruntime | onnxruntime_purego |
|-------------|--------------------|
| 1.24        | v1.24.0            |
| 1.23        | v1.23.0            |


```shell
# 下载最新版本
go get -u github.com/getcharzp/onnxruntime_purego

# 针对 onnxruntime 1.24 下载特定版本 purego
go get -u github.com/getcharzp/onnxruntime_purego@v1.24
```

## 快速开始

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

	inputData := make([]float32, 3*640*640)
	inputValue, err := engine.NewTensor([]int64{1, 3, 640, 640}, inputData)
	if err != nil {
		log.Fatal(err)
	}
	defer inputValue.Destroy()

	inputs := map[string]*ort.Value{
		"images": inputValue,
	}

	outputs, err := session.Run(inputs)
	if err != nil {
		log.Fatal(err)
	}

	for name, output := range outputs {
		outputData, err := ort.GetTensorData[float32](output)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("%v: %+v", name, outputData[:min(len(outputData), 20)])
	}
}

```

## 案例

### YOLOv11 目标检测

| 原图                                                  | Mask图                                                      |
|-----------------------------------------------------|------------------------------------------------------------|
| <img width="100%" src="./testdata/test.png" alt=""> | <img width="100%" src="./testdata/yolov11_det.png" alt=""> |

