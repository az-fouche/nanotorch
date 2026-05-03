#include "storage.h"

// Storage

Storage::Storage(py::ssize_t n, Dtype dtype, Device device) :
    buffer_(make_buffer(n * dtype_itemsize(dtype), device)),
    dtype_(dtype),
    n_(n) {}

std::shared_ptr<Storage> Storage::allocate(py::ssize_t n, Dtype dtype, Device device) {
    return std::make_shared<Storage>(n, dtype, device);
}

std::shared_ptr<Storage> Storage::from_iterable(py::sequence values, Dtype dtype, Device device) {
    py::size_t size = py::len(values);
    auto storage = Storage::allocate(size, dtype, device);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto *data_loc = static_cast<T*>(storage->data());
        for (py::size_t i = 0; i < size; ++i)
            data_loc[i] = values[i].cast<T>();
    });
    return (device == Device::Cpu) ? storage : storage->to(device);
}

std::shared_ptr<Storage> Storage::clone() const {
    auto new_storage = Storage::allocate(size(), dtype(), device());
    size_t nbytes = static_cast<size_t>(size()) * itemsize();
    if (device() == Device::Cpu)
        std::memcpy(new_storage->data(), data(), nbytes);
    else
        NT_CUDA_CHECK(
            cudaMemcpy(new_storage->data(), data(), nbytes, cudaMemcpyDeviceToDevice)
        );
    return new_storage;
}


std::shared_ptr<Storage> Storage::to(Device target_device) const {
    if (device() == target_device)
        return std::const_pointer_cast<Storage>(shared_from_this());
    auto new_storage = Storage::allocate(size(), dtype(), target_device);
    size_t nbytes = static_cast<size_t>(size()) * itemsize();
    cudaMemcpyKind kind = (device() == Device::Cpu) ? 
        cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    NT_CUDA_CHECK(cudaMemcpy(new_storage->data(), data(), nbytes, kind));
    return new_storage;
}

std::shared_ptr<Storage> Storage::cast(Dtype target) const {
    if (target == dtype()) // No copy on same dtype
        return std::const_pointer_cast<Storage>(shared_from_this());
    if (device() == Device::Cuda)
        return to(Device::Cpu)->cast(target)->to(Device::Cuda); // FIXME(#11)
    auto new_storage = Storage::allocate(size(), target, device());
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

void bind_storage_(py::module_& m) {
    py::enum_<Dtype>(m, "Dtype", py::arithmetic())
        .value("Bool", Dtype::Bool)
        .value("Int32", Dtype::Int32)
        .value("Int64", Dtype::Int64)
        .value("Float32", Dtype::Float32)
        .value("Float64", Dtype::Float64);
    py::enum_<Device>(m, "Device")
        .value("Cpu", Device::Cpu)
        .value("Cuda", Device::Cuda);
    py::class_<Storage, std::shared_ptr<Storage>>(m, "Storage", py::buffer_protocol())
        .def_static("allocate", &Storage::allocate, py::arg("n"), py::arg("dtype"), py::arg("device"))
        .def_static("from_iterable", &Storage::from_iterable, py::arg("values"), py::arg("dtype"), py::arg("device"))
        .def("clone", &Storage::clone)
        .def("to", &Storage::to, py::arg("device"))
        .def_property_readonly("size", &Storage::size)
        .def_property_readonly("itemsize", &Storage::itemsize)
        .def_property_readonly("dtype", &Storage::dtype)
        .def_property_readonly("device", &Storage::device)
        .def_buffer([](Storage& s) { return s.buffer_info(); });
}