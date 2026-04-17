#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

template<typename T>
double _sum_typed(const T* data, py::ssize_t length) {
    double result = 0.0;
    for (py::ssize_t i = 0; i < length; ++i) {
        result += data[i];
    }
    return result;
}

double sum(py::buffer buffer) {
    py::buffer_info const info = buffer.request();

    if (info.ndim != 1)
        throw std::invalid_argument(
            "sum: expected a 1D buffer, got " + std::to_string(info.ndim) + "D."
        );

    if (info.format == "f") 
        return _sum_typed(static_cast<const float*>(info.ptr), info.size);
    else if (info.format == "d") 
        return _sum_typed(static_cast<const double*>(info.ptr), info.size);
    else if (info.format == "q") 
        return _sum_typed(static_cast<const int64_t*>(info.ptr), info.size);
    else if (info.format == "i") 
        return _sum_typed(static_cast<const int32_t*>(info.ptr), info.size);
    else if (info.format == "b") 
        return _sum_typed(static_cast<const int8_t*>(info.ptr), info.size);
    
    throw std::invalid_argument(
        "sum: unsupported dtype format '" + info.format + "'."
    );
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "nanotorch C++ core module.";
    m.def("sum", &sum, "Sum all elements in a 1D buffer", py::arg("buffer"));
}