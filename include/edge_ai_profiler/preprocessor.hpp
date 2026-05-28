#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace edge_ai_profiler {

enum class PreprocessMode;

struct LetterboxInfo {
    cv::Size original_size;
    cv::Size input_size;
    cv::Size resized_size;
    float scale{1.0F};
    int pad_left{0};
    int pad_top{0};
};

struct PreprocessResult {
    std::vector<float> tensor;
    std::array<int64_t, 4> shape{};
    LetterboxInfo letterbox;
};

class ImagePreprocessor final {
public:
    ImagePreprocessor(int input_width,
                      int input_height,
                      PreprocessMode mode,
                      int openmp_threads);

    PreprocessResult Run(const cv::Mat& bgr_image) const;
    PreprocessMode mode() const { return mode_; }

private:
    int input_width_;
    int input_height_;
    PreprocessMode mode_;
    int openmp_threads_;
};

}  // namespace edge_ai_profiler
