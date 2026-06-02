#include "edge_ai_profiler/preprocessor.hpp"

#include "edge_ai_profiler/cli_options.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace edge_ai_profiler {

namespace {

void ConvertBgrToRgbNchwScalar(const cv::Mat& bgr_image,
                               std::vector<float>& tensor,
                               const int width,
                               const int height)
{
    const int plane_size = width * height;

    for (int y = 0; y < height; ++y) {
        const auto* row = bgr_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const int hw_index = y * width + x;
            const cv::Vec3b& pixel = row[x];

            tensor[static_cast<std::size_t>(hw_index)] =
                static_cast<float>(pixel[2]) / 255.0F;
            tensor[static_cast<std::size_t>(plane_size + hw_index)] =
                static_cast<float>(pixel[1]) / 255.0F;
            tensor[static_cast<std::size_t>(2 * plane_size + hw_index)] =
                static_cast<float>(pixel[0]) / 255.0F;
        }
    }
}

void ConvertBgrToRgbNchwOpenMP(const cv::Mat& bgr_image,
                               std::vector<float>& tensor,
                               const int width,
                               const int height,
                               const int thread_count)
{
    const int plane_size = width * height;
    const auto write_row = [&](const int y) {
        const auto* row = bgr_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const int hw_index = y * width + x;
            const cv::Vec3b& pixel = row[x];

            tensor[static_cast<std::size_t>(hw_index)] =
                static_cast<float>(pixel[2]) / 255.0F;
            tensor[static_cast<std::size_t>(plane_size + hw_index)] =
                static_cast<float>(pixel[1]) / 255.0F;
            tensor[static_cast<std::size_t>(2 * plane_size + hw_index)] =
                static_cast<float>(pixel[0]) / 255.0F;
        }
    };

#if EDGE_AI_PROFILER_HAS_OPENMP
    if (thread_count > 0) {
#pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int y = 0; y < height; ++y) {
            write_row(y);
        }
        return;
    }

#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        write_row(y);
    }
#else
    for (int y = 0; y < height; ++y) {
        write_row(y);
    }
#endif
}

}  // namespace

ImagePreprocessor::ImagePreprocessor(const int input_width,
                                     const int input_height,
                                     const PreprocessMode mode,
                                     const int openmp_threads)
    : input_width_(input_width),
      input_height_(input_height),
      mode_(mode),
      openmp_threads_(openmp_threads)
{
    if (input_width_ <= 0 || input_height_ <= 0) {
        throw std::runtime_error("Preprocessor input size must be positive.");
    }
    if (openmp_threads_ < 0) {
        throw std::runtime_error("OpenMP thread count must be non-negative.");
    }
#if !EDGE_AI_PROFILER_HAS_OPENMP
    if (mode_ == PreprocessMode::OpenMP) {
        throw std::runtime_error("OpenMP preprocessing was requested, but OpenMP is not available.");
    }
#endif
}

PreprocessResult ImagePreprocessor::Run(const cv::Mat& bgr_image) const
{
    if (bgr_image.empty()) {
        throw std::runtime_error("Cannot preprocess an empty image.");
    }

    cv::Mat normalized_bgr;
    if (bgr_image.channels() == 1) {
        cv::cvtColor(bgr_image, normalized_bgr, cv::COLOR_GRAY2BGR);
    } else if (bgr_image.channels() == 4) {
        cv::cvtColor(bgr_image, normalized_bgr, cv::COLOR_BGRA2BGR);
    } else if (bgr_image.channels() == 3) {
        normalized_bgr = bgr_image;
    } else {
        throw std::runtime_error("Unsupported image channel count.");
    }

    const cv::Size original_size = normalized_bgr.size();
    const float scale = std::min(
        static_cast<float>(input_width_) / static_cast<float>(original_size.width),
        static_cast<float>(input_height_) / static_cast<float>(original_size.height));

    const cv::Size resized_size(
        std::max(1, static_cast<int>(std::round(static_cast<float>(original_size.width) * scale))),
        std::max(1, static_cast<int>(std::round(static_cast<float>(original_size.height) * scale))));

    cv::Mat resized;
    cv::resize(normalized_bgr, resized, resized_size, 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat letterboxed(
        input_height_,
        input_width_,
        CV_8UC3,
        cv::Scalar(114, 114, 114));

    const int pad_left = (input_width_ - resized_size.width) / 2;
    const int pad_top = (input_height_ - resized_size.height) / 2;
    resized.copyTo(letterboxed(cv::Rect(pad_left, pad_top, resized_size.width, resized_size.height)));

    PreprocessResult result;
    result.shape = {1, 3, input_height_, input_width_};
    result.letterbox = LetterboxInfo{
        original_size,
        cv::Size(input_width_, input_height_),
        resized_size,
        scale,
        pad_left,
        pad_top};

    const int plane_size = input_width_ * input_height_;
    result.tensor.resize(static_cast<std::size_t>(3 * plane_size));

    if (mode_ == PreprocessMode::OpenMP) {
        const int thread_count = openmp_threads_ > 0 ? openmp_threads_ : 0;
        ConvertBgrToRgbNchwOpenMP(letterboxed, result.tensor, input_width_, input_height_, thread_count);
    } else {
        // Why we bypassed OpenMP here: for 640x640 tensors the conversion loop is
        // often memory-bandwidth-bound, and thread scheduling can cost more than
        // the saved arithmetic. Scalar mode is the deterministic baseline.
        ConvertBgrToRgbNchwScalar(letterboxed, result.tensor, input_width_, input_height_);
    }

    return result;
}

}  // namespace edge_ai_profiler
