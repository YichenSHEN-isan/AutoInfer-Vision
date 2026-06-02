#include "edge_ai_profiler/report_writer.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace edge_ai_profiler {
namespace {

void EnsureParentDirectory(const std::filesystem::path& path)
{
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::string JsonEscape(const std::string& value)
{
    std::ostringstream escaped;
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            escaped << "\\\\";
            break;
        case '"':
            escaped << "\\\"";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            escaped << ch;
            break;
        }
    }
    return escaped.str();
}

std::string CsvEscape(const std::string& value)
{
    bool needs_quotes = false;
    for (const char ch : value) {
        if (ch == '"' || ch == ',' || ch == '\n' || ch == '\r') {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes) {
        return value;
    }

    std::ostringstream escaped;
    escaped << '"';
    for (const char ch : value) {
        if (ch == '"') {
            escaped << "\"\"";
        } else {
            escaped << ch;
        }
    }
    escaped << '"';
    return escaped.str();
}

void WriteShapeJson(std::ostream& out, const std::vector<int64_t>& shape)
{
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        out << shape[i];
        if (i + 1 < shape.size()) {
            out << ", ";
        }
    }
    out << ']';
}

void WriteStatsJson(std::ostream& out, const char* name, const StageStats& stats, const bool trailing_comma)
{
    out << "    \"" << name << "\": {\n"
        << "      \"mean_ms\": " << stats.mean_ms << ",\n"
        << "      \"p50_ms\": " << stats.p50_ms << ",\n"
        << "      \"p95_ms\": " << stats.p95_ms << ",\n"
        << "      \"min_ms\": " << stats.min_ms << ",\n"
        << "      \"max_ms\": " << stats.max_ms << "\n"
        << "    }" << (trailing_comma ? "," : "") << "\n";
}

void WriteCsvValue(std::ostream& out, const std::string& value)
{
    out << CsvEscape(value);
}

void WriteCsvValue(std::ostream& out, const double value)
{
    out << std::fixed << std::setprecision(6) << value;
}

void WriteCsvValue(std::ostream& out, const int value)
{
    out << value;
}

template <typename T>
void WriteCsvField(std::ostream& out, const T& value, const bool last = false)
{
    WriteCsvValue(out, value);
    if (!last) {
        out << ',';
    }
}

bool NeedsHeader(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        return true;
    }
    return std::filesystem::file_size(path) == 0;
}

}  // namespace

void WriteJsonReport(const std::filesystem::path& path, const BenchmarkExport& data)
{
    EnsureParentDirectory(path);

    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open JSON report for writing: " + path.string());
    }

    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"tool\": \"edge_ai_profiler\",\n";
    out << "  \"model\": \"" << JsonEscape(data.model_path.string()) << "\",\n";
    out << "  \"source\": \"" << JsonEscape(data.source_name) << "\",\n";
    out << "  \"provider\": \"" << JsonEscape(data.provider_name) << "\",\n";
    out << "  \"preprocess_mode\": \"" << JsonEscape(data.preprocess_mode) << "\",\n";
    out << "  \"warmup_runs\": " << data.warmup_runs << ",\n";
    out << "  \"iterations\": " << data.iterations << ",\n";
    out << "  \"input_size\": [" << data.input_width << ", " << data.input_height << "],\n";
    out << "  \"cuda_device_id\": " << data.cuda_device_id << ",\n";
    out << "  \"openmp_threads\": " << data.openmp_threads << ",\n";
    out << "  \"confidence_threshold\": " << data.confidence_threshold << ",\n";
    out << "  \"nms_threshold\": " << data.nms_threshold << ",\n";
    out << "  \"input_node\": \"" << JsonEscape(data.input_name) << "\",\n";
    out << "  \"output_node\": \"" << JsonEscape(data.output_name) << "\",\n";
    out << "  \"input_shape\": ";
    WriteShapeJson(out, data.input_shape);
    out << ",\n";
    out << "  \"output_shape\": ";
    WriteShapeJson(out, data.output_shape);
    out << ",\n";
    out << "  \"benchmark\": {\n";
    WriteStatsJson(out, "preprocess", data.report.preprocess, true);
    WriteStatsJson(out, "inference", data.report.inference, true);
    WriteStatsJson(out, "postprocess", data.report.postprocess, true);
    WriteStatsJson(out, "total", data.report.total, true);
    out << "    \"throughput_fps\": " << data.report.throughput_fps << ",\n";
    out << "    \"sample_count\": " << data.report.sample_count << "\n";
    out << "  },\n";
    out << "  \"last_frame\": {\n";
    out << "    \"detection_count\": " << data.last_frame.detections.size() << ",\n";
    out << "    \"render_ms\": " << data.last_frame.timings.render_ms << ",\n";
    out << "    \"detections\": [\n";
    for (std::size_t i = 0; i < data.last_frame.detections.size(); ++i) {
        const Detection& detection = data.last_frame.detections[i];
        out << "      {\n";
        out << "        \"class_id\": " << detection.class_id << ",\n";
        out << "        \"label\": \"" << JsonEscape(detection.label) << "\",\n";
        out << "        \"confidence\": " << detection.confidence << ",\n";
        out << "        \"box\": {\"x\": " << detection.box.x
            << ", \"y\": " << detection.box.y
            << ", \"width\": " << detection.box.width
            << ", \"height\": " << detection.box.height << "}\n";
        out << "      }" << (i + 1 < data.last_frame.detections.size() ? "," : "") << "\n";
    }
    out << "    ]\n";
    out << "  }\n";
    out << "}\n";
}

void AppendCsvReport(const std::filesystem::path& path, const BenchmarkExport& data)
{
    EnsureParentDirectory(path);
    const bool write_header = NeedsHeader(path);

    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Failed to open CSV report for writing: " + path.string());
    }

    if (write_header) {
        out << "model,source,provider,preprocess_mode,warmup_runs,iterations,input_width,input_height,"
               "cuda_device_id,openmp_threads,confidence_threshold,nms_threshold,"
               "preprocess_mean_ms,preprocess_p50_ms,preprocess_p95_ms,"
               "inference_mean_ms,inference_p50_ms,inference_p95_ms,"
               "postprocess_mean_ms,postprocess_p50_ms,postprocess_p95_ms,"
               "total_mean_ms,total_p50_ms,total_p95_ms,total_min_ms,total_max_ms,"
               "throughput_fps,detection_count\n";
    }

    WriteCsvField(out, data.model_path.string());
    WriteCsvField(out, data.source_name);
    WriteCsvField(out, data.provider_name);
    WriteCsvField(out, data.preprocess_mode);
    WriteCsvField(out, data.warmup_runs);
    WriteCsvField(out, data.iterations);
    WriteCsvField(out, data.input_width);
    WriteCsvField(out, data.input_height);
    WriteCsvField(out, data.cuda_device_id);
    WriteCsvField(out, data.openmp_threads);
    WriteCsvField(out, static_cast<double>(data.confidence_threshold));
    WriteCsvField(out, static_cast<double>(data.nms_threshold));
    WriteCsvField(out, data.report.preprocess.mean_ms);
    WriteCsvField(out, data.report.preprocess.p50_ms);
    WriteCsvField(out, data.report.preprocess.p95_ms);
    WriteCsvField(out, data.report.inference.mean_ms);
    WriteCsvField(out, data.report.inference.p50_ms);
    WriteCsvField(out, data.report.inference.p95_ms);
    WriteCsvField(out, data.report.postprocess.mean_ms);
    WriteCsvField(out, data.report.postprocess.p50_ms);
    WriteCsvField(out, data.report.postprocess.p95_ms);
    WriteCsvField(out, data.report.total.mean_ms);
    WriteCsvField(out, data.report.total.p50_ms);
    WriteCsvField(out, data.report.total.p95_ms);
    WriteCsvField(out, data.report.total.min_ms);
    WriteCsvField(out, data.report.total.max_ms);
    WriteCsvField(out, data.report.throughput_fps);
    WriteCsvField(out, static_cast<int>(data.last_frame.detections.size()), true);
    out << '\n';
}

}  // namespace edge_ai_profiler
