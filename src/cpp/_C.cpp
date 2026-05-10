#include "factory.h"
#include "ops/binary_ops.h"
#include "ops/copy_ops.h"
#include "ops/cuda.h"
#include "ops/inplace_ops.h"
#include "ops/matmul.h"
#include "ops/reduction_ops.h"
#include "ops/unary_ops.h"
#include "storage.h"
#include "tensor_view.h"

PYBIND11_MODULE(_C, m) {
  m.doc() = "nanotorch C++ core module.";
  bind_storage_(m);
  bind_tensor_view_(m);
  bind_factory_(m);
  bind_ops_(m);
  bind_cuda_(m);
  bind_binary_ops_(m);
  bind_inplace_ops_(m);
  bind_matmul_op_(m);
  bind_unary_ops_(m);
  bind_reduction_ops_(m);
}