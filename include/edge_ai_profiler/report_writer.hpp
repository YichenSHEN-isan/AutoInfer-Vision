#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "edge_ai_profiler/benchmark.hpp"
#include "edge_ai_profiler/detection.hpp"

namespace edge_ai_profiler {

struct BenchmarkExport {
    std::filesystem::path model_path;
    std::string source_name;
    std::string provider_name;
    std::string preprocess_mode;
    int warmup_runs{0};
    int iterations{0};
    int input_width{0};
    int input_height{0};
    int cuda_device_id{0};
    int openmp_threads{0};
    float confidence_threshold{0.0F};
    float nms_threshold{0.0F};
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    BenchmarkReport report;
    FrameResult last_frame;
};

void WriteJsonReport(const std::filesystem::path& path, const BenchmarkExport& data);
void AppendCsvReport(const std::filesystem::path& path, const BenchmarkExport& data);

}  // namespace edge_ai_profiler
