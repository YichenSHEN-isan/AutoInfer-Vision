#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "edge_ai_profiler/detection.hpp"

namespace edge_ai_profiler {

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual void Render(const cv::Mat& frame, const FrameResult& result) = 0;
};

class OpenCVRenderer final : public IRenderer {
public:
    explicit OpenCVRenderer(std::string window_name = "Edge AI Profiler");

    void Render(const cv::Mat& frame, const FrameResult& result) override;

private:
    std::string window_name_;
};

class NullRenderer final : public IRenderer {
public:
    void Render(const cv::Mat& frame, const FrameResult& result) override;
};

}  // namespace edge_ai_profiler
