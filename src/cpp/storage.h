#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(_MSC_VER) && !defined(__clang__)
    #define NT_UNREACHABLE() __assume(false)
#elif defined(__GNUC__) || defined(__clang__)
    #define NT_UNREACHABLE() __builtin__unreachable()
#else
    #define NT_UNREACHABLE() std::abort()
#endif

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
    // Holds a contiguous memory slice
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

class Scalar {
    // Holds a scalar value with flexible type
public:
    Scalar() = default;
    Scalar(py::object obj) {
        if (py::isinstance<py::bool_>(obj))       data_ = obj.cast<bool>();
        else if (py::isinstance<py::int_>(obj))   data_ = obj.cast<int64_t>();
        else if (py::isinstance<py::float_>(obj)) data_ = obj.cast<double>();
        else throw py::type_error("Unknown scalar type.");
    }

    template <typename T>
    T item() const { return std::visit([](auto v) { return static_cast<T>(v); }, data_); }
    Dtype dtype() const { 
        switch (data_.index()) { 
            case 0: return Dtype::Bool; 
            case 1: return Dtype::Int64; 
            case 2: return Dtype::Float64;
            default: NT_UNREACHABLE();
        } 
    }
private:
    std::variant<bool, int64_t, double> data_;
};

namespace pybind11 { namespace detail {

// Registers a special Python -> Scalar caster
template <> struct type_caster<Scalar> {
    PYBIND11_TYPE_CASTER(Scalar, const_name("bool | int | float"));

    bool load(handle src, bool /*convert*/) {
        if (!src) return false;
        if (py::isinstance<py::bool_>(src)
            || py::isinstance<py::int_>(src)
            || py::isinstance<py::float_>(src)) {
            value = Scalar(reinterpret_borrow<py::object>(src));
            return true;
        }
        return false;
    }
};

}} // namespace pybind11::detail

void bind_storage_(py::module_& m);