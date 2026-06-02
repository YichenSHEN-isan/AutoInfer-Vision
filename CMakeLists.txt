cmake_minimum_required(VERSION 3.20)

project(EdgeAiProfiler
    VERSION 0.1.0
    DESCRIPTION "Lightweight edge AI performance probe and visualizer for ONNX Runtime"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EDGE_AI_PROFILER_ENABLE_OPENMP "Enable OpenMP support for preprocessing experiments" ON)

set(EDGE_AI_PROFILER_THIRD_PARTY_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    CACHE PATH "Path to third-party binary dependencies"
)

set(OpenCV_DIR
    "${EDGE_AI_PROFILER_THIRD_PARTY_DIR}/opencv/build/x64/vc16/lib"
    CACHE PATH "Path to OpenCVConfig.cmake"
)

set(ONNXRUNTIME_ROOT
    "${EDGE_AI_PROFILER_THIRD_PARTY_DIR}/onnxruntime"
    CACHE PATH "Path to ONNX Runtime binary package"
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)

file(GLOB EDGE_AI_PROFILER_OPENCV_RUNTIME_DLLS
    "${EDGE_AI_PROFILER_THIRD_PARTY_DIR}/opencv/build/x64/vc16/bin/opencv_world*.dll"
    "${EDGE_AI_PROFILER_THIRD_PARTY_DIR}/opencv/build/x64/vc16/bin/opencv_videoio_ffmpeg*.dll"
)

file(GLOB EDGE_AI_PROFILER_ONNXRUNTIME_PROVIDER_DLLS
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
)

include(cmake/onnxruntime.cmake)

if(EDGE_AI_PROFILER_ENABLE_OPENMP)
    find_package(OpenMP)
endif()

add_executable(edge_ai_profiler
    src/benchmark.cpp
    src/cli_options.cpp
    src/inference_engine.cpp
    src/main.cpp
    src/postprocessor.cpp
    src/preprocessor.cpp
    src/report_writer.cpp
    src/renderer.cpp
)

target_include_directories(edge_ai_profiler
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(edge_ai_profiler
    PRIVATE
        ${OpenCV_LIBS}
        onnxruntime::onnxruntime
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(edge_ai_profiler PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(edge_ai_profiler PRIVATE EDGE_AI_PROFILER_HAS_OPENMP=1)
else()
    target_compile_definitions(edge_ai_profiler PRIVATE EDGE_AI_PROFILER_HAS_OPENMP=0)
endif()

if(MSVC)
    target_compile_options(edge_ai_profiler PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(edge_ai_profiler PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(TARGET edge_ai_profiler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:onnxruntime::onnxruntime>"
        "$<TARGET_FILE_DIR:edge_ai_profiler>"
)

foreach(ort_provider_dll IN LISTS EDGE_AI_PROFILER_ONNXRUNTIME_PROVIDER_DLLS)
    add_custom_command(TARGET edge_ai_profiler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ort_provider_dll}"
            "$<TARGET_FILE_DIR:edge_ai_profiler>"
    )
endforeach()

foreach(opencv_dll IN LISTS EDGE_AI_PROFILER_OPENCV_RUNTIME_DLLS)
    add_custom_command(TARGET edge_ai_profiler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${opencv_dll}"
            "$<TARGET_FILE_DIR:edge_ai_profiler>"
    )
endforeach()
