#include "factory.h"

#include "cuda.cuh"

template <class T>
__global__ void _fill_kernel(py::ssize_t n, T value, T *ptr_out) {
  auto i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n)
    return;
  ptr_out[i] = value;
}

template <class T>
std::shared_ptr<Storage> _full(py::ssize_t n, T value, Dtype dtype,
                               Device device) {
  auto storage_out = Storage::allocate(n, dtype, device);
  auto ptr_out = static_cast<T *>(storage_out->data());
  switch (device) {
  case Device::Cpu:
    for (py::ssize_t i = 0; i < n; ++i)
      ptr_out[i] = value;
    break;
  case Device::Cuda:
    launch_1d(n, _fill_kernel<T>, n, value, ptr_out);
    break;
  default:
    NT_UNREACHABLE();
  }
  return storage_out;
}

std::shared_ptr<Storage> zeros(py::ssize_t n, Dtype dtype, Device device) {
  return DispatchAll::run(dtype, [&]<class T>() {
    return _full<T>(n, static_cast<T>(0), dtype, device);
  });
}

std::shared_ptr<Storage> ones(py::ssize_t n, Dtype dtype, Device device) {
  return DispatchAll::run(dtype, [&]<class T>() {
    return _full<T>(n, static_cast<T>(1), dtype, device);
  });
}

std::shared_ptr<Storage> full(py::ssize_t n, Scalar value, Dtype dtype,
                              Device device) {
  return DispatchAll::run(dtype, [&]<class T>() {
    return _full<T>(n, value.item<T>(), dtype, device);
  });
}

std::shared_ptr<Storage> eye(py::ssize_t n, Dtype dtype, Device device) {
  py::ssize_t size = n * n;
  auto storage = zeros(size, dtype, Device::Cpu);
  DispatchAll::run(dtype, [&]<class T>() {
    auto data = static_cast<T *>(storage->data());
    for (py::ssize_t i = 0; i < n; ++i)
      data[i + i * n] = static_cast<T>(1);
  });
  return storage->to(device);
}

std::shared_ptr<Storage> arange(py::ssize_t n, py::ssize_t start,
                                py::ssize_t step, Dtype dtype, Device device) {
  auto storage = Storage::allocate(n, dtype, Device::Cpu);
  DispatchAll::run(dtype, [&]<class T>() {
    auto data = static_cast<T *>(storage->data());
    for (py::ssize_t i = 0; i < n; ++i)
      data[i] = static_cast<T>(start + i * step);
  });
  return storage->to(device);
}

std::shared_ptr<Storage> uniform(py::ssize_t n, Dtype dtype, Device device) {
  auto storage = Storage::allocate(n, dtype, Device::Cpu);
  DispatchFloat::run(dtype, [&]<class T>() {
    auto data = static_cast<T *>(storage->data());
    auto &rng = global_rng();
    std::uniform_real_distribution<T> distribution(0.0, 1.0);
    for (py::ssize_t i = 0; i < n; ++i)
      data[i] = static_cast<T>(distribution(rng));
  });
  return storage->to(device);
}

void bind_factory_(py::module_ &m) {
  m.def("zeros", &zeros, "Initialize a zeros-filled vector", py::arg("shape"),
        py::arg("dtype"), py::arg("device"));
  m.def("ones", &ones, "Initialize a ones-filled vector", py::arg("shape"),
        py::arg("dtype"), py::arg("device"));
  m.def("full", &full, "Initialize a filled vector", py::arg("shape"),
        py::arg("value"), py::arg("dtype"), py::arg("device"));
  m.def("eye", &eye, "Initialize an eye matrix.", py::arg("n"),
        py::arg("dtype"), py::arg("device"));
  m.def("arange", &arange, "Initialize a range vector", py::arg("n"),
        py::arg("start"), py::arg("step"), py::arg("dtype"), py::arg("device"));
  m.def("manual_seed", &manual_seed, "Set RNG seed.", py::arg("seed"));
  m.def("uniform", &uniform, "Initialize a uniform (0, 1) vector", py::arg("n"),
        py::arg("dtype"), py::arg("device"));
}