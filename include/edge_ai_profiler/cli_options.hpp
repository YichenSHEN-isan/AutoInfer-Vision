#pragma once

#include <filesystem>
#include <optional>

namespace edge_ai_profiler {

enum class ExecutionProvider {
    Cpu,
    Cuda
};

enum class PreprocessMode {
    Scalar,
    OpenMP
};

struct AppConfig {
    std::filesystem::path model_path{"models/yolov8n.onnx"};
    std::optional<std::filesystem::path> image_path;
    std::optional<std::filesystem::path> json_report_path;
    std::optional<std::filesystem::path> csv_report_path;
    ExecutionProvider provider{ExecutionProvider::Cpu};
    PreprocessMode preprocess_mode{PreprocessMode::Scalar};
    bool display_window{false};
    int input_width{640};
    int input_height{640};
    int warmup_runs{3};
    int iterations{30};
    int cuda_device_id{0};
    int openmp_threads{0};
    float confidence_threshold{0.25F};
    float nms_threshold{0.45F};
};

class HelpRequested final {
};

AppConfig ParseCommandLine(int argc, char** argv);
void PrintUsage();
const char* ToString(ExecutionProvider provider);
const char* ToString(PreprocessMode mode);

}  // namespace edge_ai_profiler
