cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(out)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(Torch_DIR /libtorch/share/cmake/Torch)
set(CMAKE_CXX_STANDARD 11)
find_package(Torch REQUIRED)

set(SOURCE_FILES inference.cpp)
add_executable(out ${SOURCE_FILES})
target_link_libraries(out  ${TORCH_LIBRARIES})

#cmake -DCMAKE_PREFIX_PATH=/libtorch ..



