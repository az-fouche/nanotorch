"""Main nanotorch module initialization point."""

from ._data_type import Dtype, bool_, float32, float64, int32, int64
from .autograd import (
    AddOp,
    ExpOp,
    LogOp,
    MatmulOp,
    MeanOp,
    MulOp,
    NegOp,
    PowOp,
    SubOp,
    SumOp,
    TrueDivOp,
    equal_op,
    greater_eq_op,
    greater_op,
)
from .core import Tensor, TensorLike
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
    negate,
    pow,
    reshape,
    subtract,
    sum,
    transpose,
)

__all__ = [
    "Dtype",
    "Tensor",
    "TensorLike",
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
    "negate",
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


def manual_seed(seed: int) -> None:
    import numpy as np  # FIXME: implement nt.rand without np

    np.random.seed(seed)


def _totensor(x: TensorLike):
    return x if isinstance(x, Tensor) else Tensor(x)


# Runtime autograd ops binding to avoid circular imports
Tensor.__add__ = lambda self, other: AddOp.apply(self, _totensor(other))
Tensor.__radd__ = lambda self, other: AddOp.apply(self, _totensor(other))
Tensor.__iadd__ = lambda self, other: AddOp.apply(self, _totensor(other), out=self)
Tensor.__mul__ = lambda self, other: MulOp.apply(self, _totensor(other))
Tensor.__rmul__ = lambda self, other: MulOp.apply(self, _totensor(other))
Tensor.__imul__ = lambda self, other: MulOp.apply(self, _totensor(other), out=self)
Tensor.__sub__ = lambda self, other: SubOp.apply(self, _totensor(other))
Tensor.__rsub__ = lambda self, other: SubOp.apply(_totensor(_totensor(other)), self)
Tensor.__isub__ = lambda self, other: SubOp.apply(self, _totensor(other), out=self)
Tensor.__truediv__ = lambda self, other: TrueDivOp.apply(self, _totensor(other))
Tensor.__rtruediv__ = lambda self, other: TrueDivOp.apply(_totensor(other), self)
Tensor.__itruediv__ = lambda self, other: TrueDivOp.apply(
    self, _totensor(other), out=self
)
Tensor.__neg__ = lambda self: NegOp.apply(self)
Tensor.__matmul__ = lambda self, other: MatmulOp.apply(self, _totensor(other))
Tensor.__pow__ = lambda self, exponent: PowOp.apply(self, _totensor(exponent))
Tensor.__rpow__ = lambda self, exponent: PowOp.apply(_totensor(exponent), self)
Tensor.__eq__ = lambda self, other: equal_op(self, _totensor(other))
Tensor.__gt__ = lambda self, other: greater_op(self, _totensor(other))
Tensor.__ge__ = lambda self, other: greater_eq_op(self, _totensor(other))
Tensor.__lt__ = lambda self, other: _totensor(other) > self
Tensor.__le__ = lambda self, other: _totensor(other) >= self
Tensor.exp = lambda self: ExpOp.apply(self)
Tensor.log = lambda self: LogOp.apply(self)
Tensor.pow = lambda self, exponent: PowOp.apply(self, exponent)
Tensor.sum = lambda self, axis=None, keepdim=False, dtype=None: SumOp.apply(
    self, axis, keepdim, dtype
)
Tensor.mean = lambda self, axis=None, keepdim=False, dtype=None: MeanOp.apply(
    self, axis, keepdim, dtype
)

Tensor.__hash__ = None  # Necessary for __eq__ semantics
