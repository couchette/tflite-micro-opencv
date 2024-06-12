cmake_minimum_required(VERSION 3.5)

project(hello_world C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

file(GLOB hello_world_sources
    "${CMAKE_CURRENT_LIST_DIR}/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/*.cc"
    "${CMAKE_CURRENT_LIST_DIR}/models/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/models/*.cc"
)

add_executable(hello_world "")

target_sources(hello_world
    PRIVATE
    "${hello_world_sources}"
)

target_include_directories(hello_world
    PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/"
)

target_link_libraries(hello_world PRIVATE tflmicro)
