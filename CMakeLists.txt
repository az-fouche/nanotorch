cmake_minimum_required(VERSION 3.15...3.29)

project(nanotorch LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_C MODULE 
    src/cpp/_C.cpp 
    src/cpp/storage.cpp
    src/cpp/ops.cpp
    src/cpp/factory.cpp
    src/cpp/cuda/cuda.cpp
    src/cpp/cuda/unary_ops.cu
)
target_include_directories(_C PRIVATE src/cpp)
target_link_libraries(_C PRIVATE CUDA::cudart)
target_compile_options(_C PRIVATE 
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wpedantic>
)

install(TARGETS _C DESTINATION nanotorch)
