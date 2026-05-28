#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace edge_ai_profiler {

enum class ExecutionProvider;

struct TensorOutput {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

struct InferenceEngineConfig {
    ExecutionProvider provider;
    int cuda_device_id{0};
    int intra_op_threads{1};
    int inter_op_threads{1};
};

class OnnxInferenceEngine final {
public:
    OnnxInferenceEngine(const std::filesystem::path& model_path,
                        const InferenceEngineConfig& config);

    TensorOutput Run(const float* input_data,
                     std::size_t input_element_count,
                     const std::array<int64_t, 4>& input_shape);

    const std::string& input_name() const { return input_name_; }
    const std::string& output_name() const { return output_name_; }
    const std::string& provider_name() const { return provider_name_; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    std::string input_name_;
    std::string output_name_;
    std::string provider_name_;
};

}  // namespace edge_ai_profiler
