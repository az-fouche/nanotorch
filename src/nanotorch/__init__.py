"""Main nanotorch module initialization point."""

from ._data_type import DataType, bool_, float32, float64, int32, int64
from .autograd import AddOp, ExpOp, LogOp, MulOp, SumOp
from .core import Tensor
from .factories import arange, eye, full, ones, tensor, zeros
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
    "arange",
    "eye",
    "full",
    "ones",
    "tensor",
    "zeros",
    "bool_",
    "int32",
    "int64",
    "float32",
    "float64",
    "add",
    "divide",
    "multiply",
    "subtract",
    "clone",
    "matmul",
    "equals",
    "flatten",
    "mean",
    "pow",
    "reshape",
    "sum",
    "transpose",
    "exp",
    "log",
]


# Runtime autograd ops binding to avoid circular imports
Tensor.__add__ = lambda self, other: AddOp.apply(self, other)
Tensor.__radd__ = lambda self, other: AddOp.apply(self, other)
Tensor.__mul__ = lambda self, other: MulOp.apply(self, other)
Tensor.__rmul__ = lambda self, other: MulOp.apply(self, other)
Tensor.exp = lambda self: ExpOp.apply(self)
Tensor.log = lambda self: LogOp.apply(self)
Tensor.sum = lambda self, axis=None, keepdim=False, dtype=None: SumOp.apply(
    self, axis, keepdim, dtype
)
