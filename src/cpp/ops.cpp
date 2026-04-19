#include "ops.h"

py::ssize_t numel_from_shape(const std::vector<py::ssize_t>& shape) {
    auto n_axes = static_cast<py::ssize_t>(shape.size());
    py::ssize_t numel = 1;
    for (py::ssize_t i = 0; i < n_axes; ++i)
        numel *= shape[i];
    return numel;
}

// Factory

std::shared_ptr<Storage> zeros(py::ssize_t n, Dtype dtype) {
    return Storage::allocate(n, dtype); // 0 by default
}

std::shared_ptr<Storage> zeros(const std::vector<py::ssize_t>& shape, Dtype dtype) {
    return zeros(numel_from_shape(shape), dtype); // 0 by default
}

std::shared_ptr<Storage> ones(py::ssize_t n, Dtype dtype) {
    auto storage = zeros(n, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < storage->size(); ++i)
            data[i] = static_cast<T>(1);
    });
    return storage;
}

std::shared_ptr<Storage> ones(const std::vector<py::ssize_t>& shape, Dtype dtype) {
    return ones(numel_from_shape(shape), dtype);
}

std::shared_ptr<Storage> full(py::ssize_t n, Scalar value, Dtype dtype) {
    auto storage = zeros(n, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        auto fill = value.item<T>();
        for (py::ssize_t i = 0; i < storage->size(); ++i)
            data[i] = fill;
    });
    return storage;
}

std::shared_ptr<Storage> full(const std::vector<py::ssize_t>& shape, Scalar value, Dtype dtype) {
    return full(numel_from_shape(shape), value, dtype);
}

std::shared_ptr<Storage> eye(py::ssize_t n, Dtype dtype) {
    py::ssize_t size = n * n;
    auto storage = zeros(size, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < n; ++i)
            data[i + i * n] = static_cast<T>(1);
    });
    return storage;
}

std::shared_ptr<Storage> arange(py::ssize_t n, py::ssize_t start, py::ssize_t step, Dtype dtype) {
    auto storage = zeros(n, dtype);
    dispatch_dtype(dtype, [&]<typename T>() {
        auto data = static_cast<T*>(storage->data());
        for (py::ssize_t i = 0; i < n; ++i)
            data[i] = static_cast<T>(start + i * step);
    });
    return storage;
}

std::shared_ptr<Storage> arange(py::ssize_t n, Dtype dtype) {
    return arange(n, 0, 1, dtype);
}

// Ops

bool equals(
    const Storage& s1, 
    const std::vector<py::ssize_t>& sh1, 
    const std::vector<py::ssize_t>& st1, 
    py::ssize_t of1,
    const Storage& s2, 
    const std::vector<py::ssize_t>& sh2, 
    const std::vector<py::ssize_t>& st2, 
    py::ssize_t of2
) {
    if (s1.dtype() != s2.dtype())
        throw std::invalid_argument(
            "equals: expected homogeneous tensors, got " + dtype_to_format(s1.dtype()) 
            +  " and " + dtype_to_format(s2.dtype()) + "."
        );
    if (sh1 != sh2)
        throw std::invalid_argument(
            "equals: expected same size tensors, got " + vec_to_string(sh1)
            +  " and " + vec_to_string(sh2) + "."
        );

    auto n_axes = static_cast<py::ssize_t>(sh1.size());
    return dispatch_dtype(s1.dtype(), [&]<typename T>() {
        auto* ptr1 = static_cast<const T*>(s1.data());
        auto* ptr2 = static_cast<const T*>(s2.data());
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel_from_shape(sh1); ++i) {
            // Current loc eq check
            py::ssize_t idx1 = of1, idx2 = of2;
            for (py::ssize_t j = 0; j < n_axes; ++j) {
                idx1 += st1[j] * loc[j];
                idx2 += st2[j] * loc[j];
            }
            if (ptr1[idx1] != ptr2[idx2]) return false;

            // Loc update
            py::ssize_t j = n_axes - 1;
            while (j >= 0) { 
                loc[j] += 1;
                if (loc[j] < sh1[j]) break;
                loc[j] = 0;
                j -= 1;
            }
        }
        return true;
    });
}

double sum(
    const Storage& storage, 
    const std::vector<py::ssize_t>& shape, 
    const std::vector<py::ssize_t>& strides, 
    py::ssize_t offset
) {
    auto n_axes = static_cast<py::ssize_t>(shape.size());
    return dispatch_dtype(storage.dtype(), [&]<typename T>() {
        auto data = static_cast<const T*>(storage.data());
        double result = 0;
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel_from_shape(shape); ++i) {
            // Add value at loc
            py::ssize_t idx = offset;
            for (py::ssize_t j = 0; j < n_axes; ++j) idx += strides[j] * loc[j];
            result += data[idx];

            // Loc update
            py::ssize_t j = n_axes - 1;
            while (j >= 0) { 
                loc[j] += 1;
                if (loc[j] < shape[j]) break;
                loc[j] = 0;
                j -= 1;
            }
        }
        return result;
    });
}

void bind_ops_(py::module_& m) {
    m.def(
        "zeros", py::overload_cast<const std::vector<py::ssize_t>&, Dtype>(&zeros), 
        "Initialize a zeros-filled vector", py::arg("shape"), py::arg("dtype")
    );
    m.def(
        "ones", py::overload_cast<const std::vector<py::ssize_t>&, Dtype>(&ones), 
        "Initialize a ones-filled vector", py::arg("shape"), py::arg("dtype")
    );
    m.def(
        "full", py::overload_cast<const std::vector<py::ssize_t>&, Scalar, Dtype>(&full), 
        "Initialize a filled vector", py::arg("shape"), py::arg("value"), py::arg("dtype")
    );
    m.def(
        "eye", py::overload_cast<py::ssize_t, Dtype>(&eye), 
        "Initialize an eye matrix.", py::arg("n"), py::arg("dtype")
    );
    m.def(
        "arange", py::overload_cast<py::ssize_t, py::ssize_t, py::ssize_t, Dtype>(&arange), 
        "Initialize a range vector", py::arg("n"), py::arg("start"), py::arg("step"), py::arg("dtype")
    );
    m.def(
        "cast", [](const Storage& s, Dtype t) { return s.cast(t); }, 
        "Cast a storage to another dtype.", py::arg("storage"), py::arg("dtype")
    );
    m.def(
        "equals", &equals, "Test the per-coef equality of two tensors.", 
        py::arg("s1"), py::arg("sh1"), py::arg("st1"), py::arg("of1"), 
        py::arg("s2"), py::arg("sh2"), py::arg("st2"), py::arg("of2")
    );
    m.def(
        "sum", &sum, "Sum all elements in a tensor.", 
        py::arg("storage"), py::arg("sh"), py::arg("st"), py::arg("of")
    );
}