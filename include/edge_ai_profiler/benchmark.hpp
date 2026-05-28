#pragma once

#include <vector>

#include "edge_ai_profiler/detection.hpp"

namespace edge_ai_profiler {

struct StageStats {
    double min_ms{0.0};
    double mean_ms{0.0};
    double p50_ms{0.0};
    double p95_ms{0.0};
    double max_ms{0.0};
};

struct BenchmarkReport {
    StageStats preprocess;
    StageStats inference;
    StageStats postprocess;
    StageStats total;
    double throughput_fps{0.0};
    int sample_count{0};
};

class BenchmarkRecorder final {
public:
    void Add(const StageTimings& timings);
    BenchmarkReport Summarize() const;

private:
    std::vector<double> preprocess_ms_;
    std::vector<double> inference_ms_;
    std::vector<double> postprocess_ms_;
    std::vector<double> total_ms_;
};

}  // namespace edge_ai_profiler
