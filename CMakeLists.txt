cmake_minimum_required(VERSION 3.10)
project(tensor-library)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(
  "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
  "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>"
)

add_executable(tensor-library main.cc Tensor.cc test.cc)
target_link_libraries(tensor-library stdc++)
