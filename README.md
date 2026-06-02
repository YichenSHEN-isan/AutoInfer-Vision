# Edge AI Profiler

Lightweight Edge AI Performance Probe & Visualizer for ONNX Runtime.

## The Problem

Deploying Python-trained vision models to edge C++ environments often lacks a small, precise, and reproducible profiling surface. Large autonomous driving or robotics stacks usually hide critical latency behind middleware, visualization, logging, and scheduling layers. Before a model is integrated into a vehicle-side perception system, engineers need a clean native C++ probe that can answer:

- How much latency is spent in image preprocessing versus neural network inference?
- How does the same ONNX model behave across CPU and CUDA execution providers?
- Does CPU-side parallel preprocessing actually improve end-to-end throughput?
- Can detections be visualized without coupling rendering logic to the inference loop?

Edge AI Profiler is a focused C++17 tool for measuring those deployment questions with ONNX Runtime, OpenCV, and a decoupled renderer interface.

## Project Scope

This repository provides a native C++ inference profiler for lightweight vision models such as YOLOv8n exported to ONNX.

Current capabilities:

- CPU and CUDA Execution Provider selection through runtime flags.
- OpenCV-based image loading and preprocessing.
- Fused BGR-to-RGB NCHW tensor conversion without an intermediate RGB image buffer.
- YOLOv8 output decoding and NMS.
- Per-stage latency reporting with mean, p50, p95, min, and max.
- Warmup iterations separated from measured iterations.
- Scalar and OpenMP preprocessing modes for controlled assessment.
- Renderer abstraction through `IRenderer` and `RenderPacket`, with `OpenCVRenderer` as the current implementation.

## Architecture

```text
Input image
    |
    v
ImagePreprocessor
    - letterbox resize
    - BGR to RGB
    - HWC uint8 interleaved to NCHW float planar tensor
    |
    v
OnnxInferenceEngine
    - ONNX Runtime C++ API
    - CPU or CUDA Execution Provider
    |
    v
YoloPostprocessor
    - output tensor decode
    - confidence filtering
    - class-wise NMS
    |
    v
IRenderer
    - OpenCVRenderer for reference visualization
    - RenderPacket carries frame, detections, timings, frame index, and timestamp
    - future Vulkan, OpenGL, or Dear ImGui renderer can be injected here
```

Key source modules:

```text
include/edge_ai_profiler/cli_options.hpp
include/edge_ai_profiler/inference_engine.hpp
include/edge_ai_profiler/preprocessor.hpp
include/edge_ai_profiler/postprocessor.hpp
include/edge_ai_profiler/renderer.hpp
include/edge_ai_profiler/benchmark.hpp

src/cli_options.cpp
src/inference_engine.cpp
src/preprocessor.cpp
src/postprocessor.cpp
src/renderer.cpp
src/benchmark.cpp
src/main.cpp
```

## Dependency Layout

Expected local layout:

```text
AutoInfer-Vision/
  models/
    yolov8n.onnx
  third_party/
    onnxruntime/
      include/
      lib/
    opencv/
      build/
        x64/vc16/lib/
        x64/vc16/bin/
```

The runtime DLLs for ONNX Runtime and OpenCV are copied next to the executable by CMake post-build commands.

## Environment

Validated local environment:

```text
OS: Windows
Compiler: MSVC 19.44 x64
CMake: 3.26.0
OpenCV: 4.13.0
ONNX Runtime: 1.20.1 GPU package
CUDA Toolkit: 12.3
cuDNN: 9.x
GPU: NVIDIA GeForce RTX 3060 Laptop
Model: YOLOv8n ONNX, input shape [1, 3, 640, 640]
```

For CUDA Execution Provider, the following DLLs must be discoverable through `PATH`:

```powershell
where.exe cudart64_12.dll
where.exe cublas64_12.dll
where.exe nvJitLink_120_0.dll
where.exe cudnn64_9.dll
```

Example PATH entries:

```text
C:\NVIDIA\CUDA\v12.3\bin
C:\NVIDIA\CUDNN\v9\bin
```

## Model Export

Python is used only as an offline export tool. The profiler runtime is a standalone C++ application.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install ultralytics onnx

mkdir models
.\.venv\Scripts\yolo.exe export model=yolov8n.pt format=onnx imgsz=640 batch=1 dynamic=False simplify=False opset=12
move yolov8n.onnx models\yolov8n.onnx
```

The exported model used by this profiler has:

```text
Input:  images  [1, 3, 640, 640]
Output: output0 [1, 84, 8400]
```

## Build

Run from an x64 Visual Studio Developer PowerShell or x64 Native Tools Command Prompt:

```powershell
cd C:\Projects_Private\AutoInfer-Vision
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Run help:

```powershell
build\Release\edge_ai_profiler.exe --help
```

## Runtime Options

```text
--model <path>          ONNX model path. Default: models/yolov8n.onnx
--image <path>          Input image path. If omitted, a synthetic frame is used.
--provider <name>       Execution provider: cpu or cuda.
--preprocess <mode>     Preprocessing mode: scalar or openmp.
--display               Render detections with OpenCV imshow.
--input-size <int>      Square model input size. Default: 640.
--warmup <int>          Warmup iterations excluded from stats.
--iterations <int>      Measured iterations.
--cuda-device <int>     CUDA device id.
--openmp-threads <n>    OpenMP threads for preprocessing. 0 means runtime default.
--conf <float>          Detection confidence threshold.
--nms <float>           IoU threshold for non-maximum suppression.
```

## Benchmark Commands

CPU baseline:

```powershell
build\Release\edge_ai_profiler.exe `
  --model models\yolov8n.onnx `
  --image third_party\opencv\sources\samples\data\messi5.jpg `
  --provider cpu `
  --preprocess scalar `
  --warmup 5 `
  --iterations 50 `
  --conf 0.25
```

CUDA baseline:

```powershell
build\Release\edge_ai_profiler.exe `
  --model models\yolov8n.onnx `
  --image third_party\opencv\sources\samples\data\messi5.jpg `
  --provider cuda `
  --preprocess scalar `
  --warmup 5 `
  --iterations 50 `
  --conf 0.25
```

CUDA with OpenMP preprocessing:

```powershell
build\Release\edge_ai_profiler.exe `
  --model models\yolov8n.onnx `
  --image third_party\opencv\sources\samples\data\messi5.jpg `
  --provider cuda `
  --preprocess openmp `
  --openmp-threads 4 `
  --warmup 5 `
  --iterations 50 `
  --conf 0.25
```

## Benchmark Results

Measured on RTX 3060 Laptop with YOLOv8n ONNX and OpenCV sample image `messi5.jpg`.

| Provider | Preprocess Mode | Preprocess Mean | Inference Mean | Total Mean | Total p95 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | Scalar, fused | 2.645 ms | 93.353 ms | 96.675 ms | 105.988 ms | 10.344 FPS |
| CUDA | Scalar, fused | 2.645 ms | 6.780 ms | 10.135 ms | 11.698 ms | 98.667 FPS |
| CUDA | OpenMP, fused, 4 threads | 2.169 ms | 6.954 ms | 9.866 ms | 11.688 ms | 101.356 FPS |

Interpretation:

- CUDA reduces YOLOv8n inference latency from roughly 93 ms to roughly 6.8 ms on the tested laptop GPU.
- After inference moves to CUDA, preprocessing becomes a visible portion of end-to-end latency.
- Fusing BGR-to-RGB color conversion with HWC-to-NCHW layout conversion removes one intermediate image buffer and one full-frame memory pass.
- OpenMP improves preprocessing mean latency further, but it remains an explicit benchmark mode because CPU scheduling behavior can affect tail latency on laptop-class hardware.

## Preprocessing Design

OpenCV images are typically:

```text
HWC + BGR + uint8 + interleaved
```

YOLO ONNX input expects:

```text
NCHW + RGB + float32 + planar contiguous
```

The profiler explicitly performs:

```text
BGR uint8 HWC
  -> letterbox resize
  -> fused BGR read, RGB reorder, normalize to [0, 1], NCHW float tensor write
```

The core tensor write pattern is:

```cpp
tensor[0 * H * W + y * W + x] = R / 255.0f;
tensor[1 * H * W + y * W + x] = G / 255.0f;
tensor[2 * H * W + y * W + x] = B / 255.0f;
```

This conversion is memory-sensitive because it reads interleaved pixels and writes three separate planar regions. The implementation fuses color conversion and layout conversion into a single pass after letterbox resize, avoiding an intermediate RGB `cv::Mat`. For 640x640 images, scalar preprocessing is the deterministic baseline. OpenMP is available as an explicit benchmark mode, not as the default behavior.

## Renderer Interface

Rendering is intentionally decoupled from the inference loop:

```cpp
struct RenderPacket {
    cv::Mat frame;
    FrameResult result;
    int64_t frame_index;
    double timestamp_ms;
    std::string source_name;
};

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual void Render(const RenderPacket& packet) = 0;
};
```

Current renderer implementations:

```text
OpenCVRenderer
NullRenderer
```

This design keeps the inference and profiling pipeline independent from the visualization backend. A future Vulkan, OpenGL, Dear ImGui, or 3D scene renderer can be injected without changing preprocessing, inference, or postprocessing code.

## Future 3D Visualization Pipeline

The current renderer displays 2D boxes and latency metadata on top of the input frame. The `RenderPacket` contract is intentionally broader than the current OpenCV implementation so that future renderers can map the same AI output into richer visual surfaces:

```text
RenderPacket
  -> OpenCVRenderer: 2D annotated frame
  -> ImGuiRenderer: live profiling dashboard
  -> OpenGL/VulkanRenderer: animated 3D scene overlay
  -> BEV-style renderer: object positions, trails, and timing HUD
```

This keeps visualization as a replaceable boundary instead of a hardcoded side effect inside the inference loop.

## Engineering Notes

- The C++ runtime does not depend on Python.
- The profiler uses fixed input shape by default to keep latency measurements stable.
- Preprocessing fuses color conversion and tensor layout conversion to reduce memory traffic.
- Warmup iterations are excluded to avoid measuring initial graph optimization, memory allocation, and CUDA context setup.
- Render time is reported separately and excluded from benchmark throughput.
- CUDA provider initialization fails fast with a diagnostic message if CUDA or cuDNN runtime DLLs are missing.

## Roadmap

- Video stream mode with rolling latency windows.
- CSV or JSON benchmark export.
- Configurable ONNX input/output metadata validation.
- Pinned host memory and I/O binding experiments for CUDA.
- Dear ImGui renderer for live latency dashboards.
- TensorRT Execution Provider comparison.
