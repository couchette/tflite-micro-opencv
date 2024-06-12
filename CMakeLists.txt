cmake_minimum_required(VERSION 3.5)

project(tflmicro C CXX)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(tflite_dir "${CMAKE_CURRENT_SOURCE_DIR}/tflm-tree/tensorflow/lite")
set(signal_dir "${CMAKE_CURRENT_SOURCE_DIR}/tflm-tree/signal")
set(tfmicro_dir "${tflite_dir}/micro")
set(tfmicro_kernels_dir "${tfmicro_dir}/kernels")

file(GLOB srcs_tflite
    "${tflite_dir}/*.h"
    "${tflite_dir}/*.c"
    "${tflite_dir}/*.cc"
)

file(GLOB srcs_micro
    "${tfmicro_dir}/*.h"
    "${tfmicro_dir}/*.cc"
    "${tfmicro_dir}/*.c"
)

file(GLOB srcs_tflite_bridge
"${tfmicro_dir}/tflite_bridge/*.h"
    "${tfmicro_dir}/tflite_bridge/*.c"
    "${tfmicro_dir}/tflite_bridge/*.cc"
)

file(GLOB srcs_kernels
    "${tfmicro_kernels_dir}/*.h"
    "${tfmicro_kernels_dir}/*.c"
    "${tfmicro_kernels_dir}/*.cc"
)



file(GLOB signal_micro_kernels
    "${signal_dir}/micro/kernels/*.h"
    "${signal_dir}/micro/kernels/*.c"
    "${signal_dir}/micro/kernels/*.cc"
)

file(GLOB signal_src
    "${signal_dir}/src/*.h"
    "${signal_dir}/src/*.c"
    "${signal_dir}/src/*.cc"
)

set(signal_srcs
    "${signal_micro_kernels}"
    "${signal_src}"
    "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_float.h"
    "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_float.cc"
    "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_int16.h"
    "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_int16.cc"
    "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_int32.h"
    "${signal_dir}/src/kiss_fft_wrappers/kiss_fft_int32.cc"
)

set(lib_srcs
    "${srcs_tflite}"
    "${srcs_micro}"
    "${srcs_kernels}"
    "${srcs_tflite_bridge}"
    "${signal_srcs}"
    "${tflite_dir}/kernels/kernel_util.h"
    "${tflite_dir}/kernels/kernel_util.cc"
    "${tflite_dir}/micro/memory_planner/greedy_memory_planner.h"
    "${tflite_dir}/micro/memory_planner/greedy_memory_planner.cc"
    "${tflite_dir}/micro/memory_planner/linear_memory_planner.h"
    "${tflite_dir}/micro/memory_planner/linear_memory_planner.cc"
    "${tflite_dir}/micro/arena_allocator/non_persistent_arena_buffer_allocator.h"
    "${tflite_dir}/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc"
    "${tflite_dir}/micro/arena_allocator/persistent_arena_buffer_allocator.h"
    "${tflite_dir}/micro/arena_allocator/persistent_arena_buffer_allocator.cc"
    "${tflite_dir}/micro/arena_allocator/recording_single_arena_buffer_allocator.h"
    "${tflite_dir}/micro/arena_allocator/recording_single_arena_buffer_allocator.cc"
    "${tflite_dir}/micro/arena_allocator/single_arena_buffer_allocator.h"
    "${tflite_dir}/micro/arena_allocator/single_arena_buffer_allocator.cc"
    "${tflite_dir}/core/c/common.h"
    "${tflite_dir}/core/c/common.cc"
    "${tflite_dir}/core/api/error_reporter.h"
    "${tflite_dir}/core/api/error_reporter.cc"
    "${tflite_dir}/core/api/flatbuffer_conversions.h"
    "${tflite_dir}/core/api/flatbuffer_conversions.cc"
    "${tflite_dir}/core/api/tensor_utils.h"
    "${tflite_dir}/core/api/tensor_utils.cc"
    "${tflite_dir}/kernels/internal/common.h"
    "${tflite_dir}/kernels/internal/common.cc"
    "${tflite_dir}/kernels/internal/quantization_util.h"
    "${tflite_dir}/kernels/internal/quantization_util.cc"
    "${tflite_dir}/kernels/internal/portable_tensor_utils.h"
    "${tflite_dir}/kernels/internal/portable_tensor_utils.cc"
    "${tflite_dir}/kernels/internal/tensor_utils.cc"
    "${tflite_dir}/kernels/internal/tensor_ctypes.h"
    "${tflite_dir}/kernels/internal/tensor_ctypes.cc"
    "${tflite_dir}/kernels/internal/reference/portable_tensor_utils.h"
    "${tflite_dir}/kernels/internal/reference/portable_tensor_utils.cc"
    "${tflite_dir}/kernels/internal/reference/comparisons.h"
    "${tflite_dir}/kernels/internal/reference/comparisons.cc"
    "${tflite_dir}/schema/schema_utils.h"
    "${tflite_dir}/schema/schema_utils.cc"
)

set(lib_include_dir
    "${CMAKE_CURRENT_LIST_DIR}/tflm-tree/"
    "${CMAKE_CURRENT_LIST_DIR}/tflm-tree/third_party/ruy"
    "${CMAKE_CURRENT_LIST_DIR}/tflm-tree/third_party/gemmlowp"
    "${CMAKE_CURRENT_LIST_DIR}/tflm-tree/third_party/kissfft"
    "${CMAKE_CURRENT_LIST_DIR}/tflm-tree/third_party/flatbuffers"
    "${CMAKE_CURRENT_LIST_DIR}/tflm-tree/third_party/flatbuffers/include"
)

add_library(tflmicro STATIC)

target_include_directories(tflmicro
    PUBLIC
    "${lib_include_dir}"
)

target_sources(tflmicro PRIVATE
    "${lib_srcs}"
)

# add_subdirectory("tflm-tree/examples/hello_world")
# add_subdirectory("tflm-tree/examples/micro_speech")
add_subdirectory("tflm-tree/examples/person_detection")