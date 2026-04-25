#include "storage.h"
#include "tensor_view.h"
#include "ops.h"

PYBIND11_MODULE(_C, m) {
    m.doc() = "nanotorch C++ core module.";
    bind_storage_(m);
    bind_tensor_view_(m);
    bind_ops_(m);
}