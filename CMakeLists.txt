cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Fredholm CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Fredholm main.cu tests.h config.h)

set_target_properties(Fredholm PROPERTIES CUDA_ARCHITECTURES "70;75")
set_property(TARGET Fredholm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(Fredholm cublas.lib cusolver.lib)

# target_compile_options(Fredholm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#     --use_fast_math
#     --optimize 3
# >)
