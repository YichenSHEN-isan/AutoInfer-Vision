#include "edge_ai_profiler/renderer.hpp"

#include <iomanip>
#include <sstream>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace edge_ai_profiler {

OpenCVRenderer::OpenCVRenderer(std::string window_name)
    : window_name_(std::move(window_name))
{
}

void OpenCVRenderer::Render(const cv::Mat& frame, const FrameResult& result)
{
    cv::Mat canvas = frame.clone();

    for (const Detection& detection : result.detections) {
        cv::rectangle(canvas, detection.box, cv::Scalar(64, 220, 64), 2);

        std::ostringstream label_stream;
        label_stream << detection.label << ' ' << std::fixed << std::setprecision(2)
                     << detection.confidence;

        const std::string label = label_stream.str();
        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const int label_top = std::max(0, detection.box.y - text_size.height - 6);

        cv::rectangle(
            canvas,
            cv::Rect(detection.box.x, label_top, text_size.width + 6, text_size.height + baseline + 6),
            cv::Scalar(64, 220, 64),
            cv::FILLED);
        cv::putText(
            canvas,
            label,
            cv::Point(detection.box.x + 3, label_top + text_size.height + 1),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(20, 20, 20),
            1,
            cv::LINE_AA);
    }

    std::ostringstream status_stream;
    status_stream << "FPS " << std::fixed << std::setprecision(1) << result.timings.fps()
                  << " | infer " << std::setprecision(2) << result.timings.inference_ms << " ms";

    cv::putText(
        canvas,
        status_stream.str(),
        cv::Point(12, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.75,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA);

    cv::imshow(window_name_, canvas);
    cv::waitKey(1);
}

void NullRenderer::Render(const cv::Mat& frame, const FrameResult& result)
{
    (void)frame;
    (void)result;
}

}  // namespace edge_ai_profiler
