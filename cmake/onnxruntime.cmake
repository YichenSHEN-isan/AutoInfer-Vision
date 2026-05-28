if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers were not found under: ${ONNXRUNTIME_ROOT}/include")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    message(FATAL_ERROR "ONNX Runtime import library was not found under: ${ONNXRUNTIME_ROOT}/lib")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
    message(FATAL_ERROR "ONNX Runtime runtime library was not found under: ${ONNXRUNTIME_ROOT}/lib")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)

set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)
