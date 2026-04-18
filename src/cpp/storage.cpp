#include "storage.h"

// Dtype

std::string dtype_to_format(Dtype dtype) {
    switch (dtype) {
        case Dtype::Bool: return std::string(FORMAT_BOOL);
        case Dtype::Int32: return std::string(FORMAT_INT32);
        case Dtype::Int64: return std::string(FORMAT_INT64);
        case Dtype::Float32: return std::string(FORMAT_FLOAT32);
        case Dtype::Float64: return std::string(FORMAT_FLOAT64);
        default: throw std::invalid_argument("Unknown dtype.");
    }
}

Dtype format_to_dtype(const std::string& format) {
    if (format == FORMAT_BOOL) return Dtype::Bool;
    if (format == FORMAT_INT32) return Dtype::Int32;
    if (format == FORMAT_INT64) return Dtype::Int64;
    if (format == FORMAT_FLOAT32) return Dtype::Float32;
    if (format == FORMAT_FLOAT64) return Dtype::Float64;
    throw std::invalid_argument("Unparsable format type: " + format);
}

py::ssize_t dtype_itemsize(Dtype dtype) {
    switch (dtype) {
        case Dtype::Bool: return SIZEOF_BOOL;
        case Dtype::Int32: return SIZEOF_INT32;
        case Dtype::Int64: return SIZEOF_INT64;
        case Dtype::Float32: return SIZEOF_FLOAT32;
        case Dtype::Float64: return SIZEOF_FLOAT64;
        default: throw std::invalid_argument("Unknown dtype.");
    }
}

// Storage

Storage::Storage(py::ssize_t n, Dtype dtype) :
    data_(n * dtype_itemsize(dtype)),
    dtype_(dtype),
    n_(n) {}

std::shared_ptr<Storage> Storage::allocate(py::ssize_t n, Dtype dtype) {
    return std::make_shared<Storage>(n, dtype);
}

std::shared_ptr<Storage> Storage::from_iterable(py::sequence values, Dtype dtype) {
    py::size_t size = py::len(values);
    auto storage = Storage::allocate(size, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto *data_loc = static_cast<T*>(storage->data());
        for (py::size_t i = 0; i < size; ++i)
            data_loc[i] = values[i].cast<T>();
    });
    return storage;
}

std::shared_ptr<Storage> Storage::clone() const {
    auto new_storage = Storage::allocate(size(), dtype());
    new_storage->data_ = data_;
    return new_storage;
}

std::shared_ptr<Storage> Storage::cast(Dtype target) const {
    if (target == dtype()) // No copy on same dtype
        return std::const_pointer_cast<Storage>(shared_from_this());
    auto new_storage = Storage::allocate(size(), target);
    dispatch_dtype(dtype(), [&]<typename Src>() {
        dispatch_dtype(target, [&]<typename Dst>() {
            auto* s = static_cast<const Src*>(data());
            auto* d = static_cast<Dst*>(new_storage->data());
            for (py::ssize_t i = 0; i < size(); ++i)
                if constexpr (std::is_same_v<Dst, bool>)
                    d[i] = s[i] ? Dst{1} : Dst{0};
                else
                    d[i] = static_cast<Dst>(s[i]);
        });
    });
    return new_storage;
}

py::buffer_info Storage::buffer_info() const {
    return py::buffer_info(
        const_cast<void*>(data()),
        itemsize(),
        dtype_to_format(dtype()),
        size()
    );
}

void bind_storage(py::module_& m) {
    py::enum_<Dtype>(m, "Dtype")
        .value("Bool", Dtype::Bool)
        .value("Int32", Dtype::Int32)
        .value("Int64", Dtype::Int64)
        .value("Float32", Dtype::Float32)
        .value("Float64", Dtype::Float64);
    py::class_<Storage, std::shared_ptr<Storage>>(m, "Storage", py::buffer_protocol())
        .def_static("allocate", &Storage::allocate, py::arg("n"), py::arg("dtype"))
        .def_static("from_iterable", &Storage::from_iterable, py::arg("values"), py::arg("dtype"))
        .def("clone", &Storage::clone)
        .def_property_readonly("size", &Storage::size)
        .def_property_readonly("itemsize", &Storage::itemsize)
        .def_property_readonly("dtype", &Storage::dtype)
        .def_buffer([](Storage& s) { return s.buffer_info(); });
}