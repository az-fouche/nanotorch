#include "matmul.h"
#include "cuda.cuh"

template <class T>
std::shared_ptr<Storage> _cpu_matmul(
    const TensorView& x1, 
    const TensorView& x2,
    py::ssize_t ndim,
    const std::vector<py::ssize_t>& out_shape,
    py::ssize_t csize
) {
    auto numel = numel_from_shape(out_shape);
    auto new_storage = Storage::allocate(numel, x1.storage->dtype(), Device::Cpu);
    auto* ptr1 = static_cast<const T*>(x1.storage->data());
    auto* ptr2 = static_cast<const T*>(x2.storage->data());
    auto prout = static_cast<T*>(new_storage->data());
    std::vector<py::ssize_t> loc(ndim);
    for (py::ssize_t prout_idx = 0; prout_idx < numel; ++prout_idx) {
        // B, i, j -> derive B, i, c and B, c, j
        T acc = static_cast<T>(0.0);
        for (int c = 0; c < csize; ++c) {
            py::ssize_t idx1 = x1.offset 
                                + c * x1.strides[ndim - 1] 
                                + loc[ndim - 2] * x1.strides[ndim - 2];
            py::ssize_t idx2 = x2.offset 
                                + c * x2.strides[ndim - 2]
                                + loc[ndim - 1] * x2.strides[ndim - 1];
            for (auto dim_i = 0; dim_i < ndim - 2; ++dim_i) {
                idx1 += x1.strides[dim_i] * loc[dim_i];
                idx2 += x2.strides[dim_i] * loc[dim_i];
            }
            acc += ptr1[idx1] * ptr2[idx2];
        }
        prout[prout_idx] = acc;

        py::ssize_t j = ndim - 1;
        while (j >= 0) { 
            loc[j] += 1;
            if (loc[j] < out_shape[j]) break;
            loc[j] = 0;
            j -= 1;
        }
    }
    return new_storage;
}

template <class T>
__global__ void _matmul_kernel(
    const T* in_a, const T* in_b, T* out, 
    py::ssize_t numel, py::ssize_t ndim, 
    py::ssize_t csize, py::ssize_t lsize, py::ssize_t rsize,
    StridedView view_a, StridedView view_b
) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    py::ssize_t loc[NT_MAX_DIMS];
    py::ssize_t r = i;
    loc[ndim - 1] = r % rsize; r /= rsize;
    loc[ndim - 2] = r % lsize; r /= lsize;
    for (int j = ndim - 3; j >= 0; --j) {
        loc[j] = r % view_a.shape[j];
        r /= view_a.shape[j];
    }

    T acc = 0;
    for (py::ssize_t c = 0; c < csize; ++c) {
        py::ssize_t idx1 = view_a.offset 
                            + c * view_a.strides[ndim - 1] 
                            + loc[ndim - 2] * view_a.strides[ndim - 2];
        py::ssize_t idx2 = view_b.offset 
                            + c * view_b.strides[ndim - 2]
                            + loc[ndim - 1] * view_b.strides[ndim - 1];
        for (auto dim_i = 0; dim_i < ndim - 2; ++dim_i) {
            idx1 += view_a.strides[dim_i] * loc[dim_i];
            idx2 += view_b.strides[dim_i] * loc[dim_i];
        }
        acc += in_a[idx1] * in_b[idx2];
    }

    out[i] = acc;
}

template <class T>
std::shared_ptr<Storage> _cuda_matmul(
    const TensorView& x1, 
    const TensorView& x2,
    py::ssize_t ndim,
    const std::vector<py::ssize_t>& out_shape,
    py::ssize_t csize,
    py::ssize_t lsize,
    py::ssize_t rsize
) {
    auto numel = numel_from_shape(out_shape);
    auto new_storage = Storage::allocate(numel, x1.storage->dtype(), Device::Cuda);
    auto* ptr1 = static_cast<const T*>(x1.storage->data());
    auto* ptr2 = static_cast<const T*>(x2.storage->data());
    auto prout = static_cast<T*>(new_storage->data());

    auto view_a = StridedView(x1.shape.size(), x1.offset);
    for (size_t j = 0; j < x1.shape.size(); ++j) {
        view_a.shape[j] = x1.shape[j];
        view_a.strides[j] = x1.strides[j];
    }
    auto view_b = StridedView(x2.shape.size(), x2.offset);
    for (size_t j = 0; j < x2.shape.size(); ++j) {
        view_b.shape[j] = x2.shape[j];
        view_b.strides[j] = x2.strides[j];
    }

    launch_1d(
        numel, _matmul_kernel<T>,
        ptr1, ptr2, prout, numel, ndim, csize, lsize, rsize, view_a, view_b
    );

    return new_storage;
}

std::shared_ptr<Storage> matmul(const TensorView& x1, const TensorView& x2) {
    // Sanity checks
    auto s1 = x1.storage;
    auto s2 = x2.storage;
    if (s1->dtype() != s2->dtype())
        throw std::invalid_argument(
            "matmul: expected homogeneous tensors, got " + dtype_to_format(s1->dtype()) 
            +  " and " + dtype_to_format(s2->dtype()) + "."
        );
    if (s1->device() != s2->device())
        throw std::invalid_argument("matmul: expected homogeneous devices.");
    if (x1.shape.size() != x2.shape.size())
        throw std::invalid_argument("matmul: x1 and x2 have different ndim.");
    if (x1.shape.size() < 2)
        throw std::invalid_argument("matmul: kernel should be called with >=2d tensors.");
        
    auto ndim = x1.shape.size();
    auto out_shape = std::vector<py::ssize_t>(ndim);
    for (auto i = 0; i < ndim - 2; ++i) {
        if (x1.shape[i] != x2.shape[i]) 
            throw std::invalid_argument("Invalid broacast shapes.");
        else out_shape[i] = x1.shape[i];
    }
    py::ssize_t lsize = x1.shape[ndim - 2];
    py::ssize_t csize = x1.shape[ndim - 1];
    py::ssize_t rsize = x2.shape[ndim - 1];
    if (csize != x2.shape[ndim - 2])
        throw std::invalid_argument("Invalid matmul shape.");
    out_shape[ndim - 2] = lsize;
    out_shape[ndim - 1] = rsize;

    return dispatch_dtype_arithmetic(s1->dtype(), [&]<typename T>() {
        switch (x1.storage->device()) {
            case Device::Cpu: return _cpu_matmul<T>(x1, x2, ndim, out_shape, csize);
            case Device::Cuda: return _cuda_matmul<T>(x1, x2, ndim, out_shape, csize, lsize, rsize);
        }
    });
}

void bind_matmul_op_(py::module_& m) {
    m.def("matmul", &matmul, "Matrix multiplication.", py::arg("x1"), py::arg("x2"));
}