#pragma once

#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

#include "edge_ai_profiler/detection.hpp"

namespace edge_ai_profiler {

struct RenderPacket {
    cv::Mat frame;
    FrameResult result;
    int64_t frame_index{0};
    double timestamp_ms{0.0};
    std::string source_name;
};

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual void Render(const RenderPacket& packet) = 0;
};

class OpenCVRenderer final : public IRenderer {
public:
    explicit OpenCVRenderer(std::string window_name = "Edge AI Profiler");

    void Render(const RenderPacket& packet) override;

private:
    std::string window_name_;
};

class NullRenderer final : public IRenderer {
public:
    void Render(const RenderPacket& packet) override;
};

}  // namespace edge_ai_profiler
