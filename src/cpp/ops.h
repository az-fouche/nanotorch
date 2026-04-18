#include <pybind11/pybind11.h>

#include "storage.h"

namespace py = pybind11;

bool equals(const Storage& s1, const Storage& s2);
double sum(const Storage& storage);

void bind_ops_(py::module_& m);