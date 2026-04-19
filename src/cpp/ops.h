#include <pybind11/pybind11.h>

#include "helpers.h"
#include "storage.h"

namespace py = pybind11;

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