from ._data_type import DataType, bool_, float32, float64, int32, int64
from .autograd import FunctionAdd, FunctionExp, FunctionLog, FunctionMul, FunctionSum
from .core import Tensor
from .factories import arange, eye, full, ones, tensor, zeros
from .ops import add, divide, matmul, multiply, subtract

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
    "matmul",
]


Tensor.__add__ = lambda self, other: FunctionAdd.apply(self, other)
Tensor.__radd__ = lambda self, other: FunctionAdd.apply(self, other)
Tensor.__mul__ = lambda self, other: FunctionMul.apply(self, other)
Tensor.__rmul__ = lambda self, other: FunctionMul.apply(self, other)
Tensor.exp = lambda self: FunctionExp.apply(self)
Tensor.log = lambda self: FunctionLog.apply(self)
Tensor.sum = lambda self, axis=None, keepdim=False, dtype=None: FunctionSum.apply(
    self, axis, keepdim, dtype
)
