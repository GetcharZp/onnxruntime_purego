#!/usr/bin/env bash
#
# 拉取测试所需、但未入库的二进制依赖：
#   1. ONNX Runtime 预编译动态库 -> lib/<DefaultLibraryPath() 约定的文件名>
#   2. yolo11n.onnx 模型          -> testdata/yolo11n.onnx
#
# 依赖已存在时自动跳过，可重复执行；本地与 CI（三平台）通用。
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.26.0}"
ORT_BASE="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"

cd "$(dirname "$0")/.."
mkdir -p lib # lib/ 已被 .gitignore 忽略，全新 clone 中并不存在

# 平台相关的资产名 + DefaultLibraryPath() 期望的目标文件名
case "$(go env GOOS)/$(go env GOARCH)" in
windows/amd64) asset="onnxruntime-win-x64-${ORT_VERSION}.zip" lib="onnxruntime.dll" ;;
linux/amd64) asset="onnxruntime-linux-x64-${ORT_VERSION}.tgz" lib="onnxruntime_amd64.so" ;;
linux/arm64) asset="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz" lib="onnxruntime_arm64.so" ;;
darwin/arm64) asset="onnxruntime-osx-arm64-${ORT_VERSION}.tgz" lib="onnxruntime_arm64.dylib" ;;
# 注：onnxruntime 自 1.x 后期起不再发布 mac x86_64 预编译包，故无 darwin/amd64 分支
*)
	echo "不支持的平台: $(go env GOOS)/$(go env GOARCH)" >&2
	exit 1
	;;
esac

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ ! -f "lib/${lib}" ]; then
	echo "==> 下载 ${asset}"
	curl -fsSL "${ORT_BASE}/${asset}" -o "${tmp}/${asset}"
	case "$asset" in
	*.zip) # Windows 压缩包：直接把 onnxruntime.dll 写到目标位置
		unzip -p "${tmp}/${asset}" '*/lib/onnxruntime.dll' >"lib/${lib}"
		;;
	*) # tgz 压缩包：libonnxruntime.* 多为符号链接，只取体积达标的实体文件（跳过符号链接与 dSYM 调试符号）
		tar -xzf "${tmp}/${asset}" -C "$tmp"
		src=$(find "$tmp" -name 'libonnxruntime.*' ! -type d ! -path '*dSYM*' -size +1M -print -quit)
		cp -Lf "$src" "lib/${lib}"
		;;
	esac
fi

if [ ! -f "testdata/yolo11n.onnx" ]; then
	echo "==> 下载 yolo11n.onnx"
	curl -fsSL "$MODEL_URL" -o testdata/yolo11n.onnx
fi

echo "==> 依赖就绪: lib/${lib}, testdata/yolo11n.onnx"
