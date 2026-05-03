#pragma once

#include <stdexcept>

#include <pybind11/pybind11.h>

namespace py = pybind11;

inline constexpr std::string_view FORMAT_BOOL = "b";
inline constexpr std::string_view FORMAT_INT32 = "i";
inline constexpr std::string_view FORMAT_INT64 = "q";
inline constexpr std::string_view FORMAT_FLOAT32 = "f";
inline constexpr std::string_view FORMAT_FLOAT64 = "d";

inline constexpr py::ssize_t SIZEOF_BOOL = 1;
inline constexpr py::ssize_t SIZEOF_INT32 = 4;
inline constexpr py::ssize_t SIZEOF_INT64 = 8;
inline constexpr py::ssize_t SIZEOF_FLOAT32 = 4;
inline constexpr py::ssize_t SIZEOF_FLOAT64 = 8;

enum class Dtype : uint8_t {
    Bool,
    Int32,
    Int64,
    Float32,
    Float64
};

template <typename F>
auto dispatch_dtype(Dtype dtype, F&& func) {
    switch (dtype) {
        case Dtype::Bool: return func.template operator()<bool>();
        case Dtype::Int32: return func.template operator()<int32_t>();
        case Dtype::Int64: return func.template operator()<int64_t>();
        case Dtype::Float32: return func.template operator()<float>();
        case Dtype::Float64: return func.template operator()<double>();
        default: throw std::invalid_argument("Unknown dtype.");
    }
}
struct DispatchAll {
    template <class F> static auto run(Dtype dt, F&& f) { 
        return dispatch_dtype(dt, std::forward<F>(f)); 
    }
};

template <typename F>
auto dispatch_dtype_float(Dtype dtype, F&& func) {
    switch (dtype) {
        case Dtype::Float32: return func.template operator()<float>();
        case Dtype::Float64: return func.template operator()<double>();
        default: throw std::invalid_argument("Invalid dtype.");
    }
}
struct DispatchFloat {
    template <class F> static auto run(Dtype dt, F&& f) { 
        return dispatch_dtype_float(dt, std::forward<F>(f)); 
    }
};

template <typename F>
auto dispatch_dtype_arithmetic(Dtype dtype, F&& func) {
    switch (dtype) {
        case Dtype::Int32: return func.template operator()<int32_t>();
        case Dtype::Int64: return func.template operator()<int64_t>();
        case Dtype::Float32: return func.template operator()<float>();
        case Dtype::Float64: return func.template operator()<double>();
        default: throw std::invalid_argument("Invalid dtype.");
    }
}
struct DispatchArithmetic {
    template <class F> static auto run(Dtype dt, F&& f) { 
        return dispatch_dtype_arithmetic(dt, std::forward<F>(f)); 
    }
};

inline std::string dtype_to_format(Dtype dtype) {
    switch (dtype) {
        case Dtype::Bool: return std::string(FORMAT_BOOL);
        case Dtype::Int32: return std::string(FORMAT_INT32);
        case Dtype::Int64: return std::string(FORMAT_INT64);
        case Dtype::Float32: return std::string(FORMAT_FLOAT32);
        case Dtype::Float64: return std::string(FORMAT_FLOAT64);
        default: throw std::invalid_argument("Unknown dtype.");
    }
}

inline Dtype format_to_dtype(const std::string& format) {
    if (format == FORMAT_BOOL) return Dtype::Bool;
    if (format == FORMAT_INT32) return Dtype::Int32;
    if (format == FORMAT_INT64) return Dtype::Int64;
    if (format == FORMAT_FLOAT32) return Dtype::Float32;
    if (format == FORMAT_FLOAT64) return Dtype::Float64;
    throw std::invalid_argument("Unparsable format type: " + format);
}

inline py::ssize_t dtype_itemsize(Dtype dtype) {
    switch (dtype) {
        case Dtype::Bool: return SIZEOF_BOOL;
        case Dtype::Int32: return SIZEOF_INT32;
        case Dtype::Int64: return SIZEOF_INT64;
        case Dtype::Float32: return SIZEOF_FLOAT32;
        case Dtype::Float64: return SIZEOF_FLOAT64;
        default: throw std::invalid_argument("Unknown dtype.");
    }
}