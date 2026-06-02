#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "edge_ai_profiler/benchmark.hpp"
#include "edge_ai_profiler/cli_options.hpp"
#include "edge_ai_profiler/inference_engine.hpp"
#include "edge_ai_profiler/postprocessor.hpp"
#include "edge_ai_profiler/preprocessor.hpp"
#include "edge_ai_profiler/report_writer.hpp"
#include "edge_ai_profiler/renderer.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMilliseconds(const Clock::time_point start, const Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

cv::Mat LoadFrame(const edge_ai_profiler::AppConfig& config)
{
    if (config.image_path.has_value()) {
        cv::Mat image = cv::imread(config.image_path->string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + config.image_path->string());
        }
        return image;
    }

    cv::Mat synthetic(480, 640, CV_8UC3, cv::Scalar(32, 36, 42));
    cv::rectangle(synthetic, cv::Rect(180, 130, 260, 180), cv::Scalar(80, 130, 210), 3);
    cv::putText(
        synthetic,
        "Synthetic frame",
        cv::Point(170, 245),
        cv::FONT_HERSHEY_SIMPLEX,
        0.9,
        cv::Scalar(230, 230, 230),
        2,
        cv::LINE_AA);
    return synthetic;
}

std::string SourceName(const edge_ai_profiler::AppConfig& config)
{
    return config.image_path.has_value() ? config.image_path->string() : "synthetic";
}

edge_ai_profiler::FrameResult RunOnce(
    const cv::Mat& frame,
    const edge_ai_profiler::ImagePreprocessor& preprocessor,
    edge_ai_profiler::OnnxInferenceEngine& engine,
    const edge_ai_profiler::YoloPostprocessor& postprocessor,
    edge_ai_profiler::PreprocessResult& input,
    edge_ai_profiler::TensorOutput& output)
{
    auto start = Clock::now();
    input = preprocessor.Run(frame);
    auto after_preprocess = Clock::now();

    output = engine.Run(input.tensor.data(), input.tensor.size(), input.shape);
    auto after_inference = Clock::now();

    edge_ai_profiler::FrameResult result;
    result.detections = postprocessor.Decode(output.data, output.shape, input.letterbox);
    auto after_postprocess = Clock::now();

    result.timings.preprocess_ms = ElapsedMilliseconds(start, after_preprocess);
    result.timings.inference_ms = ElapsedMilliseconds(after_preprocess, after_inference);
    result.timings.postprocess_ms = ElapsedMilliseconds(after_inference, after_postprocess);
    result.timings.total_ms =
        result.timings.preprocess_ms + result.timings.inference_ms + result.timings.postprocess_ms;

    return result;
}

void PrintStageStats(const char* label, const edge_ai_profiler::StageStats& stats)
{
    std::cout << std::left << std::setw(14) << label << std::right
              << std::setw(10) << stats.mean_ms
              << std::setw(10) << stats.p50_ms
              << std::setw(10) << stats.p95_ms
              << std::setw(10) << stats.min_ms
              << std::setw(10) << stats.max_ms << '\n';
}

void PrintReport(const edge_ai_profiler::AppConfig& config,
                 const edge_ai_profiler::OnnxInferenceEngine& engine,
                 const edge_ai_profiler::FrameResult& result,
                 const edge_ai_profiler::PreprocessResult& input,
                 const edge_ai_profiler::TensorOutput& output,
                 const edge_ai_profiler::BenchmarkReport& report)
{
    std::cout << "\n=== Edge AI Profiler Report ===\n";
    std::cout << "Model: " << config.model_path.string() << '\n';
    std::cout << "Provider: " << engine.provider_name() << '\n';
    std::cout << "Preprocess mode: " << edge_ai_profiler::ToString(config.preprocess_mode) << '\n';
    if (config.preprocess_mode == edge_ai_profiler::PreprocessMode::OpenMP) {
        std::cout << "OpenMP preprocessing threads: ";
        if (config.openmp_threads > 0) {
            std::cout << config.openmp_threads << '\n';
        } else {
            std::cout << "runtime default\n";
        }
    }
    std::cout << "Warmup runs: " << config.warmup_runs << '\n';
    std::cout << "Measured iterations: " << report.sample_count << '\n';
    std::cout << "Input node: " << engine.input_name() << '\n';
    std::cout << "Output node: " << engine.output_name() << '\n';
    std::cout << "Input tensor: [" << input.shape[0] << ", " << input.shape[1] << ", "
              << input.shape[2] << ", " << input.shape[3] << "]\n";
    std::cout << "Output tensor: [";
    for (std::size_t i = 0; i < output.shape.size(); ++i) {
        std::cout << output.shape[i] << (i + 1 < output.shape.size() ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nStage latency in milliseconds. Render is excluded from benchmark throughput.\n";
    std::cout << std::left << std::setw(14) << "Stage" << std::right
              << std::setw(10) << "mean"
              << std::setw(10) << "p50"
              << std::setw(10) << "p95"
              << std::setw(10) << "min"
              << std::setw(10) << "max" << '\n';
    PrintStageStats("preprocess", report.preprocess);
    PrintStageStats("inference", report.inference);
    PrintStageStats("postprocess", report.postprocess);
    PrintStageStats("total", report.total);
    std::cout << "\nThroughput: " << report.throughput_fps << " FPS\n";
    std::cout << "Last-frame detections: " << result.detections.size() << '\n';
    std::cout << "Last render time: " << result.timings.render_ms << " ms\n";

    for (const auto& detection : result.detections) {
        std::cout << " - " << detection.label << " score=" << detection.confidence
                  << " box=(" << detection.box.x << ", " << detection.box.y << ", "
                  << detection.box.width << ", " << detection.box.height << ")\n";
    }
}

std::vector<int64_t> ToVector(const std::array<int64_t, 4>& shape)
{
    return {shape.begin(), shape.end()};
}

edge_ai_profiler::BenchmarkExport BuildExportData(
    const edge_ai_profiler::AppConfig& config,
    const edge_ai_profiler::OnnxInferenceEngine& engine,
    const edge_ai_profiler::FrameResult& result,
    const edge_ai_profiler::PreprocessResult& input,
    const edge_ai_profiler::TensorOutput& output,
    const edge_ai_profiler::BenchmarkReport& report)
{
    return edge_ai_profiler::BenchmarkExport{
        config.model_path,
        SourceName(config),
        engine.provider_name(),
        edge_ai_profiler::ToString(config.preprocess_mode),
        config.warmup_runs,
        config.iterations,
        config.input_width,
        config.input_height,
        config.cuda_device_id,
        config.openmp_threads,
        config.confidence_threshold,
        config.nms_threshold,
        engine.input_name(),
        engine.output_name(),
        ToVector(input.shape),
        output.shape,
        report,
        result};
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const edge_ai_profiler::AppConfig config =
            edge_ai_profiler::ParseCommandLine(argc, argv);

        cv::Mat frame = LoadFrame(config);

        edge_ai_profiler::OnnxInferenceEngine engine(
            config.model_path,
            edge_ai_profiler::InferenceEngineConfig{
                config.provider,
                config.cuda_device_id,
                1,
                1});
        edge_ai_profiler::ImagePreprocessor preprocessor(
            config.input_width,
            config.input_height,
            config.preprocess_mode,
            config.openmp_threads);
        edge_ai_profiler::YoloPostprocessor postprocessor({
            config.confidence_threshold,
            config.nms_threshold,
            100});

        std::unique_ptr<edge_ai_profiler::IRenderer> renderer;
        if (config.display_window) {
            renderer = std::make_unique<edge_ai_profiler::OpenCVRenderer>();
        } else {
            renderer = std::make_unique<edge_ai_profiler::NullRenderer>();
        }

        edge_ai_profiler::PreprocessResult input;
        edge_ai_profiler::TensorOutput output;
        edge_ai_profiler::FrameResult result;

        for (int i = 0; i < config.warmup_runs; ++i) {
            result = RunOnce(frame, preprocessor, engine, postprocessor, input, output);
        }

        edge_ai_profiler::BenchmarkRecorder recorder;
        for (int i = 0; i < config.iterations; ++i) {
            result = RunOnce(frame, preprocessor, engine, postprocessor, input, output);
            recorder.Add(result.timings);
        }

        const edge_ai_profiler::BenchmarkReport report = recorder.Summarize();

        auto before_render = Clock::now();
        renderer->Render(edge_ai_profiler::RenderPacket{
            frame,
            result,
            0,
            0.0,
            SourceName(config)});
        auto after_render = Clock::now();

        result.timings.render_ms = ElapsedMilliseconds(before_render, after_render);

        PrintReport(config, engine, result, input, output, report);
        const edge_ai_profiler::BenchmarkExport export_data =
            BuildExportData(config, engine, result, input, output, report);

        if (config.json_report_path.has_value()) {
            edge_ai_profiler::WriteJsonReport(*config.json_report_path, export_data);
            std::cout << "JSON report written: " << config.json_report_path->string() << '\n';
        }
        if (config.csv_report_path.has_value()) {
            edge_ai_profiler::AppendCsvReport(*config.csv_report_path, export_data);
            std::cout << "CSV row appended: " << config.csv_report_path->string() << '\n';
        }
        return 0;
    } catch (const edge_ai_profiler::HelpRequested&) {
        edge_ai_profiler::PrintUsage();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        std::cerr << "Run with --help to see supported options.\n";
        return 1;
    }
}
