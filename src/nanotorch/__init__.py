"""Main nanotorch module initialization point."""

from ._data_type import DataType, bool_, float32, float64, int32, int64
from .autograd import (
    AddOp,
    ExpOp,
    LogOp,
    MatmulOp,
    MeanOp,
    MulOp,
    PowOp,
    SubOp,
    SumOp,
)
from .core import Tensor
from .factories import arange, eye, full, ones, rand, tensor, zeros
from .ops import (
    add,
    clone,
    divide,
    equals,
    exp,
    flatten,
    log,
    matmul,
    mean,
    multiply,
    pow,
    reshape,
    subtract,
    sum,
    transpose,
)

__all__ = [
    "DataType",
    "Tensor",
    "add",
    "arange",
    "bool_",
    "clone",
    "divide",
    "equals",
    "exp",
    "eye",
    "flatten",
    "float32",
    "float64",
    "full",
    "int32",
    "int64",
    "log",
    "matmul",
    "mean",
    "multiply",
    "ones",
    "pow",
    "rand",
    "reshape",
    "subtract",
    "sum",
    "tensor",
    "transpose",
    "zeros",
]

if not sorted(__all__) == __all__:
    raise ImportError("__all__ should be sorted.")

# Runtime autograd ops binding to avoid circular imports
Tensor.__add__ = lambda self, other: AddOp.apply(self, other)
Tensor.__radd__ = lambda self, other: AddOp.apply(self, other)
Tensor.__mul__ = lambda self, other: MulOp.apply(self, other)
Tensor.__rmul__ = lambda self, other: MulOp.apply(self, other)
Tensor.__sub__ = lambda self, other: SubOp.apply(self, other)  # TODO: rsub
Tensor.__matmul__ = lambda self, other: MatmulOp.apply(self, other)
Tensor.exp = lambda self: ExpOp.apply(self)
Tensor.log = lambda self: LogOp.apply(self)
Tensor.pow = lambda self, exponent: PowOp.apply(self, exponent)
Tensor.sum = lambda self, axis=None, keepdim=False, dtype=None: SumOp.apply(
    self, axis, keepdim, dtype
)
Tensor.mean = lambda self, axis=None, keepdim=False, dtype=None: MeanOp.apply(
    self, axis, keepdim, dtype
)
