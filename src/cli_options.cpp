#include "edge_ai_profiler/cli_options.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace edge_ai_profiler {
namespace {

std::string RequireValue(int& index, int argc, char** argv, const std::string& option)
{
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for option: " + option);
    }
    ++index;
    return argv[index];
}

ExecutionProvider ParseExecutionProvider(const std::string& value)
{
    if (value == "cpu") {
        return ExecutionProvider::Cpu;
    }
    if (value == "cuda") {
        return ExecutionProvider::Cuda;
    }
    throw std::runtime_error("Unsupported provider: " + value);
}

PreprocessMode ParsePreprocessMode(const std::string& value)
{
    if (value == "scalar") {
        return PreprocessMode::Scalar;
    }
    if (value == "openmp") {
        return PreprocessMode::OpenMP;
    }
    throw std::runtime_error("Unsupported preprocessing mode: " + value);
}

}  // namespace

AppConfig ParseCommandLine(int argc, char** argv)
{
    AppConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];

        if (option == "--help" || option == "-h") {
            throw HelpRequested{};
        }
        if (option == "--model") {
            config.model_path = RequireValue(i, argc, argv, option);
        } else if (option == "--image") {
            config.image_path = RequireValue(i, argc, argv, option);
        } else if (option == "--report") {
            config.json_report_path = RequireValue(i, argc, argv, option);
        } else if (option == "--csv") {
            config.csv_report_path = RequireValue(i, argc, argv, option);
        } else if (option == "--provider") {
            config.provider = ParseExecutionProvider(RequireValue(i, argc, argv, option));
        } else if (option == "--preprocess") {
            config.preprocess_mode = ParsePreprocessMode(RequireValue(i, argc, argv, option));
        } else if (option == "--display") {
            config.display_window = true;
        } else if (option == "--input-size") {
            const int size = std::stoi(RequireValue(i, argc, argv, option));
            config.input_width = size;
            config.input_height = size;
        } else if (option == "--warmup") {
            config.warmup_runs = std::stoi(RequireValue(i, argc, argv, option));
        } else if (option == "--iterations") {
            config.iterations = std::stoi(RequireValue(i, argc, argv, option));
        } else if (option == "--cuda-device") {
            config.cuda_device_id = std::stoi(RequireValue(i, argc, argv, option));
        } else if (option == "--openmp-threads") {
            config.openmp_threads = std::stoi(RequireValue(i, argc, argv, option));
        } else if (option == "--conf") {
            config.confidence_threshold = std::stof(RequireValue(i, argc, argv, option));
        } else if (option == "--nms") {
            config.nms_threshold = std::stof(RequireValue(i, argc, argv, option));
        } else {
            throw std::runtime_error("Unknown option: " + option);
        }
    }

    if (config.input_width <= 0 || config.input_height <= 0) {
        throw std::runtime_error("Input size must be positive.");
    }
    if (config.warmup_runs < 0) {
        throw std::runtime_error("Warmup runs must be non-negative.");
    }
    if (config.iterations <= 0) {
        throw std::runtime_error("Iterations must be positive.");
    }
    if (config.cuda_device_id < 0) {
        throw std::runtime_error("CUDA device id must be non-negative.");
    }
    if (config.openmp_threads < 0) {
        throw std::runtime_error("OpenMP thread count must be non-negative.");
    }
    if (config.confidence_threshold < 0.0F || config.confidence_threshold > 1.0F) {
        throw std::runtime_error("Confidence threshold must be in [0, 1].");
    }
    if (config.nms_threshold < 0.0F || config.nms_threshold > 1.0F) {
        throw std::runtime_error("NMS threshold must be in [0, 1].");
    }

    return config;
}

void PrintUsage()
{
    std::cout
        << "Usage:\n"
        << "  edge_ai_profiler --model models/yolov8n.onnx [--image path] [options]\n\n"
        << "Options:\n"
        << "  --model <path>       ONNX model path. Default: models/yolov8n.onnx\n"
        << "  --image <path>       Input image path. If omitted, a synthetic frame is used.\n"
        << "  --report <path>      Write a JSON benchmark report.\n"
        << "  --csv <path>         Append one benchmark row to a CSV file.\n"
        << "  --provider <name>    Execution provider: cpu or cuda. Default: cpu\n"
        << "  --preprocess <mode>  Preprocessing mode: scalar or openmp. Default: scalar\n"
        << "  --display            Render detections with OpenCV imshow.\n"
        << "  --input-size <int>   Square model input size. Default: 640\n"
        << "  --warmup <int>       Warmup iterations excluded from stats. Default: 3\n"
        << "  --iterations <int>   Measured iterations. Default: 30\n"
        << "  --cuda-device <int>  CUDA device id when --provider cuda is used. Default: 0\n"
        << "  --openmp-threads <n> OpenMP threads for preprocessing. 0 means runtime default.\n"
        << "  --conf <float>       Detection confidence threshold. Default: 0.25\n"
        << "  --nms <float>        IoU threshold for non-maximum suppression. Default: 0.45\n"
        << "  --help, -h           Show this help message.\n";
}

const char* ToString(const ExecutionProvider provider)
{
    switch (provider) {
    case ExecutionProvider::Cpu:
        return "cpu";
    case ExecutionProvider::Cuda:
        return "cuda";
    }
    return "unknown";
}

const char* ToString(const PreprocessMode mode)
{
    switch (mode) {
    case PreprocessMode::Scalar:
        return "scalar";
    case PreprocessMode::OpenMP:
        return "openmp";
    }
    return "unknown";
}

}  // namespace edge_ai_profiler
