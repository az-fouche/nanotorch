cmake_minimum_required(VERSION 3.15...3.29)

project(nanotorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_C MODULE src/cpp/_C.cpp)

install(TARGETS _C DESTINATION nanotorch)

