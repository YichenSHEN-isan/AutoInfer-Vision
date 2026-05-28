#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace edge_ai_profiler {

struct Detection {
    int class_id{-1};
    std::string label;
    float confidence{0.0F};
    cv::Rect box;
};

struct StageTimings {
    double preprocess_ms{0.0};
    double inference_ms{0.0};
    double postprocess_ms{0.0};
    double render_ms{0.0};
    double total_ms{0.0};

    double fps() const
    {
        return total_ms > 0.0 ? 1000.0 / total_ms : 0.0;
    }
};

struct FrameResult {
    std::vector<Detection> detections;
    StageTimings timings;
};

}  // namespace edge_ai_profiler
