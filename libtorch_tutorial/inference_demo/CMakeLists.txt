cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(Torch_DIR /libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(out inference_classification.cpp)
target_link_libraries(out ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET out PROPERTY CXX_STANDARD 11)

