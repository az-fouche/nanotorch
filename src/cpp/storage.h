#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

std::string dtype_to_format(Dtype dtype);
Dtype format_to_dtype(const std::string& format);
py::ssize_t dtype_itemsize(Dtype dtype);

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

class Storage : public std::enable_shared_from_this<Storage> {
public:
    Storage(py::ssize_t n, Dtype dtype);
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) = default;
    Storage& operator=(Storage&&) = default;

    // Ops
    static std::shared_ptr<Storage> allocate(py::ssize_t n, Dtype dtype);
    static std::shared_ptr<Storage> from_iterable(py::sequence values, Dtype dtype);
    std::shared_ptr<Storage> clone() const;
    std::shared_ptr<Storage> cast(Dtype target) const;
    py::buffer_info buffer_info() const;

    void* data() { return data_.data(); }
    const void* data() const { return data_.data(); }
    py::ssize_t size() const { return n_; }
    py::ssize_t itemsize() const { return dtype_itemsize(dtype_); }
    Dtype dtype() const { return dtype_; }

private:
    std::vector<char> data_;
    Dtype dtype_;
    py::ssize_t n_;
};

void bind_storage_(py::module_& m);