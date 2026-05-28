#include "edge_ai_profiler/postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace edge_ai_profiler {
namespace {

std::vector<std::string> CocoLabels()
{
    return {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};
}

struct Candidate {
    int class_id{-1};
    float confidence{0.0F};
    cv::Rect2f box;
};

float IntersectionOverUnion(const cv::Rect2f& lhs, const cv::Rect2f& rhs)
{
    const float x1 = std::max(lhs.x, rhs.x);
    const float y1 = std::max(lhs.y, rhs.y);
    const float x2 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float y2 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);

    const float intersection_width = std::max(0.0F, x2 - x1);
    const float intersection_height = std::max(0.0F, y2 - y1);
    const float intersection_area = intersection_width * intersection_height;
    const float union_area = lhs.area() + rhs.area() - intersection_area;

    return union_area > 0.0F ? intersection_area / union_area : 0.0F;
}

cv::Rect ToClampedRect(const cv::Rect2f& box, const cv::Size& image_size)
{
    const float x1 = std::clamp(box.x, 0.0F, static_cast<float>(image_size.width - 1));
    const float y1 = std::clamp(box.y, 0.0F, static_cast<float>(image_size.height - 1));
    const float x2 = std::clamp(box.x + box.width, 0.0F, static_cast<float>(image_size.width - 1));
    const float y2 = std::clamp(box.y + box.height, 0.0F, static_cast<float>(image_size.height - 1));

    const int left = static_cast<int>(std::round(x1));
    const int top = static_cast<int>(std::round(y1));
    const int right = static_cast<int>(std::round(x2));
    const int bottom = static_cast<int>(std::round(y2));

    return cv::Rect(left, top, std::max(1, right - left), std::max(1, bottom - top));
}

}  // namespace

YoloPostprocessor::YoloPostprocessor(PostprocessConfig config)
    : config_(config),
      labels_(CocoLabels())
{
}

std::vector<Detection> YoloPostprocessor::Decode(const std::vector<float>& output_data,
                                                 const std::vector<int64_t>& output_shape,
                                                 const LetterboxInfo& letterbox) const
{
    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] < 5) {
        throw std::runtime_error("Expected YOLO output shape [1, attributes, candidates].");
    }

    const int64_t attributes = output_shape[1];
    const int64_t candidates = output_shape[2];
    const int64_t class_count = attributes - 4;
    const std::size_t expected_elements =
        static_cast<std::size_t>(attributes * candidates);

    if (output_data.size() != expected_elements) {
        throw std::runtime_error("YOLO output data size does not match output shape.");
    }

    std::vector<Candidate> raw_candidates;
    raw_candidates.reserve(static_cast<std::size_t>(candidates));

    for (int64_t i = 0; i < candidates; ++i) {
        int best_class = -1;
        float best_score = 0.0F;

        for (int64_t class_index = 0; class_index < class_count; ++class_index) {
            const float score = output_data[static_cast<std::size_t>((4 + class_index) * candidates + i)];
            if (score > best_score) {
                best_score = score;
                best_class = static_cast<int>(class_index);
            }
        }

        if (best_score < config_.confidence_threshold) {
            continue;
        }

        const float center_x = output_data[static_cast<std::size_t>(0 * candidates + i)];
        const float center_y = output_data[static_cast<std::size_t>(1 * candidates + i)];
        const float width = output_data[static_cast<std::size_t>(2 * candidates + i)];
        const float height = output_data[static_cast<std::size_t>(3 * candidates + i)];

        const float x1 = (center_x - width * 0.5F - static_cast<float>(letterbox.pad_left)) / letterbox.scale;
        const float y1 = (center_y - height * 0.5F - static_cast<float>(letterbox.pad_top)) / letterbox.scale;
        const float x2 = (center_x + width * 0.5F - static_cast<float>(letterbox.pad_left)) / letterbox.scale;
        const float y2 = (center_y + height * 0.5F - static_cast<float>(letterbox.pad_top)) / letterbox.scale;

        raw_candidates.push_back(Candidate{
            best_class,
            best_score,
            cv::Rect2f(x1, y1, std::max(1.0F, x2 - x1), std::max(1.0F, y2 - y1))});
    }

    std::vector<int> order(raw_candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&raw_candidates](const int lhs, const int rhs) {
        return raw_candidates[static_cast<std::size_t>(lhs)].confidence >
               raw_candidates[static_cast<std::size_t>(rhs)].confidence;
    });

    std::vector<bool> suppressed(raw_candidates.size(), false);
    std::vector<Detection> detections;

    for (const int index : order) {
        const auto candidate_index = static_cast<std::size_t>(index);
        if (suppressed[candidate_index]) {
            continue;
        }

        const Candidate& candidate = raw_candidates[candidate_index];
        const std::string label =
            candidate.class_id >= 0 && candidate.class_id < static_cast<int>(labels_.size())
                ? labels_[static_cast<std::size_t>(candidate.class_id)]
                : "class_" + std::to_string(candidate.class_id);

        detections.push_back(Detection{
            candidate.class_id,
            label,
            candidate.confidence,
            ToClampedRect(candidate.box, letterbox.original_size)});

        if (static_cast<int>(detections.size()) >= config_.max_detections) {
            break;
        }

        for (std::size_t j = 0; j < raw_candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (raw_candidates[j].class_id != candidate.class_id) {
                continue;
            }
            if (IntersectionOverUnion(candidate.box, raw_candidates[j].box) > config_.nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return detections;
}

}  // namespace edge_ai_profiler
