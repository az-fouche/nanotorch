#include <pybind11/pybind11.h>

#include "helpers.h"
#include "storage.h"

namespace py = pybind11;

// Factory

std::shared_ptr<Storage> zeros(py::ssize_t n, Dtype dtype);
std::shared_ptr<Storage> zeros(const std::vector<py::ssize_t>& shape, Dtype dtype);
std::shared_ptr<Storage> ones(py::ssize_t n, Dtype dtype);
std::shared_ptr<Storage> ones(const std::vector<py::ssize_t>& shape, Dtype dtype);
std::shared_ptr<Storage> full(py::ssize_t n, Scalar value, Dtype dtype);
std::shared_ptr<Storage> full(const std::vector<py::ssize_t>& shape, Scalar value, Dtype dtype);
std::shared_ptr<Storage> eye(py::ssize_t n, Dtype dtype);
std::shared_ptr<Storage> arange(py::ssize_t n, py::ssize_t start, py::ssize_t step, Dtype dtype);
std::shared_ptr<Storage> arange(py::ssize_t n, Dtype dtype);

// Core ops
bool equals(
    const Storage& s1, 
        const std::vector<py::ssize_t>& sh1, // Shape
        const std::vector<py::ssize_t>& st1, // Strides
        py::ssize_t of1, // Offset
    const Storage& s2, 
        const std::vector<py::ssize_t>& sh2, 
        const std::vector<py::ssize_t>& st2, 
        py::ssize_t of2
);
double sum(
    const Storage& storage, 
    const std::vector<py::ssize_t>& shape, 
    const std::vector<py::ssize_t>& strides, 
    py::ssize_t offset
);

void bind_ops_(py::module_& m);