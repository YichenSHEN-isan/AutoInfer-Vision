#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "edge_ai_profiler/detection.hpp"
#include "edge_ai_profiler/preprocessor.hpp"

namespace edge_ai_profiler {

struct PostprocessConfig {
    float confidence_threshold{0.25F};
    float nms_threshold{0.45F};
    int max_detections{100};
};

class YoloPostprocessor final {
public:
    explicit YoloPostprocessor(PostprocessConfig config);

    std::vector<Detection> Decode(const std::vector<float>& output_data,
                                  const std::vector<int64_t>& output_shape,
                                  const LetterboxInfo& letterbox) const;

private:
    PostprocessConfig config_;
    std::vector<std::string> labels_;
};

}  // namespace edge_ai_profiler
