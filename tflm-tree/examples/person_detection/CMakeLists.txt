cmake_minimum_required(VERSION 3.5)

project(person_detection C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/Program Files/OpenCV/opencv/build/x64/MinGW64/bin")
find_package(OpenCV REQUIRED)

file(GLOB person_detection_sources
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.cc"
  "${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/lite/micro/models/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/lite/micro/models/*.cc"
)

add_executable(person_detection "")

target_sources(person_detection
  PRIVATE
  "${person_detection_sources}"
)

target_include_directories(person_detection
  PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
  "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(person_detection PRIVATE tflmicro ${OpenCV_LIBS})
