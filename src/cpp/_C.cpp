#include <pybind11/pybind11.h>

#include "storage.h"

namespace py = pybind11;

bool equals(const Storage& s1, const Storage& s2) {
    if (s1.dtype() != s2.dtype())
        throw std::invalid_argument(
            "equals: expected homogeneous storages, got " + dtype_to_format(s1.dtype()) 
            +  " and " + dtype_to_format(s2.dtype()) + "."
        );
    if (s1.size() != s2.size())
        throw std::invalid_argument(
            "equals: expected same size storages, got " + dtype_to_format(s1.dtype()) 
            +  " and " + dtype_to_format(s2.dtype()) + "."
        );
    return dispatch_dtype(s1.dtype(), [&]<typename T>() {
        auto* ptr1 = static_cast<const T*>(s1.data());
        auto* ptr2 = static_cast<const T*>(s2.data());
        for (py::ssize_t i = 0; i < s1.size(); ++i)
            if (ptr1[i] != ptr2[i]) return false;
        return true;
    });
}

double sum(const Storage& storage) {
    return dispatch_dtype(storage.dtype(), [&]<typename T>() {
        auto data = static_cast<const T*>(storage.data());
        double result = 0;
        for (py::ssize_t i = 0; i < storage.size(); ++i)
            result += data[i];
        return result;
    });
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "nanotorch C++ core module.";
    m.def(
        "cast", [](const Storage& s, Dtype t) { return s.cast(t); }, 
        "Cast a storage to another dtype.", py::arg("storage"), py::arg("dtype")
    );
    m.def("equals", &equals, "Test the per-coef equality of two storages.", py::arg("s1"), py::arg("s2"));
    m.def("sum", &sum, "Sum all elements in a 1D storages", py::arg("storage"));
    bind_storage(m);
}