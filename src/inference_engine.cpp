#include "edge_ai_profiler/inference_engine.hpp"

#include "edge_ai_profiler/cli_options.hpp"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace edge_ai_profiler {
namespace {

#ifdef _WIN32
std::wstring ToOrtPath(const std::filesystem::path& path)
{
    return std::filesystem::absolute(path).wstring();
}
#else
std::string ToOrtPath(const std::filesystem::path& path)
{
    return std::filesystem::absolute(path).string();
}
#endif

std::size_t ShapeElementCount(const std::vector<int64_t>& shape)
{
    return static_cast<std::size_t>(
        std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
}

}  // namespace

OnnxInferenceEngine::OnnxInferenceEngine(const std::filesystem::path& model_path,
                                         const InferenceEngineConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "edge_ai_profiler"),
      session_options_{},
      session_{nullptr}
{
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path.string());
    }

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(config.intra_op_threads);
    session_options_.SetInterOpNumThreads(config.inter_op_threads);

    provider_name_ = ToString(config.provider);

    try {
        if (config.provider == ExecutionProvider::Cuda) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config.cuda_device_id;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
        }

        const auto ort_model_path = ToOrtPath(model_path);
        session_ = Ort::Session(env_, ort_model_path.c_str(), session_options_);
    } catch (const Ort::Exception& error) {
        std::ostringstream message;
        message << "Failed to initialize ONNX Runtime provider '" << provider_name_
                << "': " << error.what();
        if (config.provider == ExecutionProvider::Cuda) {
            message << "\nCUDA EP requires onnxruntime_providers_cuda.dll plus matching "
                       "CUDA and cuDNN runtime DLLs on PATH. For this ORT package, verify "
                       "that cudnn64_9.dll and the CUDA 12 runtime DLLs are discoverable.";
        }
        throw std::runtime_error(message.str());
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_.GetInputNameAllocated(0, allocator);
    auto output_name = session_.GetOutputNameAllocated(0, allocator);
    input_name_ = input_name.get();
    output_name_ = output_name.get();
}

TensorOutput OnnxInferenceEngine::Run(const float* input_data,
                                      const std::size_t input_element_count,
                                      const std::array<int64_t, 4>& input_shape)
{
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data),
        input_element_count,
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    if (output_tensors.empty() || !output_tensors.front().IsTensor()) {
        throw std::runtime_error("ONNX Runtime returned no tensor output.");
    }

    auto& output_tensor = output_tensors.front();
    const auto shape_info = output_tensor.GetTensorTypeAndShapeInfo();
    TensorOutput output;
    output.shape = shape_info.GetShape();
    const std::size_t output_element_count = ShapeElementCount(output.shape);

    const float* output_data = output_tensor.GetTensorData<float>();
    output.data.assign(output_data, output_data + output_element_count);

    return output;
}

}  // namespace edge_ai_profiler
