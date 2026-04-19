#include "ops.h"

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
    py::ssize_t numel_total = 1;
    for (py::ssize_t i = 0; i < n_axes; ++i)
        numel_total *= sh1[i];

    return dispatch_dtype(s1.dtype(), [&]<typename T>() {
        auto* ptr1 = static_cast<const T*>(s1.data());
        auto* ptr2 = static_cast<const T*>(s2.data());
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel_total; ++i) {
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
    py::ssize_t numel_total = 1;
    for (py::ssize_t i = 0; i < n_axes; ++i)
        numel_total *= shape[i];

    return dispatch_dtype(storage.dtype(), [&]<typename T>() {
        auto data = static_cast<const T*>(storage.data());
        double result = 0;
        std::vector<py::ssize_t> loc(n_axes);
        for (py::ssize_t i = 0; i < numel_total; ++i) {
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