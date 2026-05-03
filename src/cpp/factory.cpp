#include "factory.h"

std::shared_ptr<Storage> zeros(py::ssize_t n, Dtype dtype, Device device) {
    return Storage::allocate(n, dtype, device); // 0 by default
}

std::shared_ptr<Storage> ones(py::ssize_t n, Dtype dtype, Device device) {
    auto storage = zeros(n, dtype, Device::Cpu); 
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < storage->size(); ++i)
            data[i] = static_cast<T>(1);
    });
    return storage->to(device);
}

std::shared_ptr<Storage> full(py::ssize_t n, Scalar value, Dtype dtype, Device device) {
    auto storage = zeros(n, dtype, Device::Cpu); // TODO(#14): Avoid device move
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        auto fill = value.item<T>();
        for (py::ssize_t i = 0; i < storage->size(); ++i)
            data[i] = fill;
    });
    return storage->to(device);
}

std::shared_ptr<Storage> eye(py::ssize_t n, Dtype dtype, Device device) {
    py::ssize_t size = n * n;
    auto storage = zeros(size, dtype, Device::Cpu);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < n; ++i)
            data[i + i * n] = static_cast<T>(1);
    });
    return storage->to(device);
}

std::shared_ptr<Storage> arange(py::ssize_t n, py::ssize_t start, py::ssize_t step, Dtype dtype, Device device) {
    auto storage = zeros(n, dtype, Device::Cpu);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < n; ++i)
            data[i] = static_cast<T>(start + i * step);
    });
    return storage->to(device);
}

// Random

namespace {
    std::mt19937_64& global_rng(){ 
        static std::mt19937_64 rng{std::random_device{}()};
        return rng;
    }
}

// FIXME: not thread-safe + doesn't affect CuRand
void manual_seed(uint64_t seed) { global_rng().seed(seed); }

std::shared_ptr<Storage> uniform(py::ssize_t n, Dtype dtype, Device device) {
    auto storage = zeros(n, dtype, Device::Cpu);
    DispatchFloat::run(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        auto& rng = global_rng();
        std::uniform_real_distribution<T> distribution(0.0, 1.0);
        for (py::ssize_t i = 0; i < n; ++i)
            data[i] = static_cast<T>(distribution(rng));
    });
    return storage->to(device);
}

void bind_factory_(py::module_& m) {
    m.def(
        "zeros", &zeros, "Initialize a zeros-filled vector", 
        py::arg("shape"), py::arg("dtype"), py::arg("device")
    );
    m.def(
        "ones", &ones, "Initialize a ones-filled vector", 
        py::arg("shape"), py::arg("dtype"), py::arg("device")
    );
    m.def(
        "full", &full, "Initialize a filled vector", 
        py::arg("shape"), py::arg("value"), py::arg("dtype"), py::arg("device")
    );
    m.def(
        "eye", &eye, "Initialize an eye matrix.", 
        py::arg("n"), py::arg("dtype"), py::arg("device")
    );
    m.def(
        "arange", &arange, "Initialize a range vector", 
        py::arg("n"), py::arg("start"), py::arg("step"), py::arg("dtype"), py::arg("device")
    );
    m.def("manual_seed", &manual_seed, "Set RNG seed.", py::arg("seed"));
    m.def(
        "uniform", &uniform, "Initialize a uniform (0, 1) vector", 
        py::arg("n"), py::arg("dtype"), py::arg("device")
    );
}