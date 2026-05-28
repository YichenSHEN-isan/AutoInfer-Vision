#include "edge_ai_profiler/benchmark.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace edge_ai_profiler {
namespace {

StageStats ComputeStats(std::vector<double> samples)
{
    if (samples.empty()) {
        throw std::runtime_error("Cannot summarize an empty benchmark sample set.");
    }

    std::sort(samples.begin(), samples.end());

    const auto percentile = [&samples](const double ratio) {
        const double position = ratio * static_cast<double>(samples.size() - 1);
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, samples.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return samples[lower] * (1.0 - fraction) + samples[upper] * fraction;
    };

    StageStats stats;
    stats.min_ms = samples.front();
    stats.max_ms = samples.back();
    stats.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) /
                    static_cast<double>(samples.size());
    stats.p50_ms = percentile(0.50);
    stats.p95_ms = percentile(0.95);
    return stats;
}

}  // namespace

void BenchmarkRecorder::Add(const StageTimings& timings)
{
    preprocess_ms_.push_back(timings.preprocess_ms);
    inference_ms_.push_back(timings.inference_ms);
    postprocess_ms_.push_back(timings.postprocess_ms);
    total_ms_.push_back(timings.total_ms);
}

BenchmarkReport BenchmarkRecorder::Summarize() const
{
    BenchmarkReport report;
    report.sample_count = static_cast<int>(total_ms_.size());
    report.preprocess = ComputeStats(preprocess_ms_);
    report.inference = ComputeStats(inference_ms_);
    report.postprocess = ComputeStats(postprocess_ms_);
    report.total = ComputeStats(total_ms_);
    report.throughput_fps =
        report.total.mean_ms > 0.0 ? 1000.0 / report.total.mean_ms : 0.0;
    return report;
}

}  // namespace edge_ai_profiler
