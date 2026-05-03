#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda/cuda.h"
#include "dtype.h"

#if defined(_MSC_VER) && !defined(__clang__)
    #define NT_UNREACHABLE() __assume(false)
#elif defined(__GNUC__) || defined(__clang__)
    #define NT_UNREACHABLE() __builtin_unreachable()
#else
    #define NT_UNREACHABLE() std::abort()
#endif

namespace py = pybind11;

enum class Device : uint8_t {
    Cpu,
    Cuda
};

// Device-agnostic memory pointer handling cleanup
struct BufDeleter {
    Device device;
    void operator()(void* p) const noexcept {
        if (!p) return;
        if (device == Device::Cuda) cudaFree(p);
        else ::operator delete(p);
    }
};
using Buffer = std::unique_ptr<void, BufDeleter>;
inline Buffer make_buffer(size_t nbytes, Device device) {
    void* p = nullptr;
    if (device == Device::Cuda) {
        cudaMalloc(&p, nbytes);
        cudaMemset(p, 0, nbytes);
    }
    else {
        p = ::operator new(nbytes);
        std::memset(p, 0, nbytes);
    }

    return Buffer(p, BufDeleter{device});
}

// 1D data container for tensors
class Storage : public std::enable_shared_from_this<Storage> {
public:
    Storage(py::ssize_t n, Dtype dtype, Device device);
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) = default;
    Storage& operator=(Storage&&) = default;

    // Ops
    static std::shared_ptr<Storage> allocate(py::ssize_t n, Dtype dtype, Device device);
    static std::shared_ptr<Storage> from_iterable(py::sequence values, Dtype dtype, Device device);
    std::shared_ptr<Storage> clone() const;
    std::shared_ptr<Storage> to(Device device) const;
    std::shared_ptr<Storage> cast(Dtype target) const;
    py::buffer_info buffer_info() const;

    void* data() { return buffer_.get(); }
    const void* data() const { return buffer_.get(); }
    Device device() const { return buffer_.get_deleter().device; }
    py::ssize_t size() const { return n_; }
    py::ssize_t itemsize() const { return dtype_itemsize(dtype_); }
    Dtype dtype() const { return dtype_; }
    uint64_t version() const { return version_; }
    void bump_version() { version_ += 1; }

private:
    Buffer buffer_;
    Dtype dtype_;
    py::ssize_t n_;
    uint64_t version_ = 0;
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
        return {};
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