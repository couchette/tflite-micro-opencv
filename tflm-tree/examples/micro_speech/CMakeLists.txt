cmake_minimum_required(VERSION 3.5)

project(micro_speech C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

file(GLOB micro_speech_sources
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.cc"
  "${CMAKE_CURRENT_LIST_DIR}/models/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/models/*.cc"
  "${CMAKE_CURRENT_LIST_DIR}/testdata/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/testdata/*.cc"
)

add_executable(micro_speech "")

target_sources(micro_speech
  PRIVATE
  "${micro_speech_sources}"
)

target_include_directories(micro_speech
  PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/"
)

target_link_libraries(micro_speech PRIVATE tflmicro)
