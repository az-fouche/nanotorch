"""Binary differentiable tensor operations and their derivative."""

import math
from typing import Callable

from nanotorch import _C  # type: ignore[missing-import]
from nanotorch._data_type import promote_dtypes
from nanotorch._indexing import broadcast_shapes
from nanotorch.core import Tensor, TensorShape, sizeof

from . import ops_spec as sp
from .function import Function
from .grad_mode import is_grad_enabled
from .ops_utils import unbroadcast


class AddOp(Function):
    """Add two tensors component-wise with broadcast.

    Output tensor y is promoted and broadcast to the best suited dtype and shape
    to prevent loss of information and ensure compatibility. Broadcasting is
    right-aligned, (a, b, c, d) :: (c, d) is broadcast to (a, b, c, d), and
    (a, b, 1, d) :: (c, d) is broadcast to (a, b, c, d).

    Parameters
    ----------
    x1: Tensor
        First term, shape S.
    x2: Tensor
        Second term, shape T.
    out: Tensor | None = None
        Optional result container to avoid new copy.

    Returns
    -------
    Tensor
        New tensor containing the coordinate-wise addition.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("tensor", sp.BroadcastableTo(0), sp.Real()),
    )

    def forward(self, x1: Tensor, x2: Tensor, out: Tensor | None = None) -> Tensor:
        self._shapes = (x1.shape, x2.shape)
        return _binary_kernel_op(x1, x2, _C.add, _C.add_inplace, out=out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return tuple(unbroadcast(grad_out, shape) for shape in self._shapes)

    @classmethod
    def flops(cls, x1: Tensor, x2: Tensor, _: Tensor | None = None) -> int:
        return math.prod(broadcast_shapes(x1.shape, x2.shape))

    @classmethod
    def mem_bytes(cls, x1: Tensor, x2: Tensor, _: Tensor | None = None) -> int:
        shape = broadcast_shapes(x1.shape, x2.shape)
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        return 3 * math.prod(shape) * sizeof(dtype)


class SubOp(Function):
    """Subtract two tensors component-wise with broadcast.

    Output tensor y is promoted and broadcast to the best suited dtype and shape
    to prevent loss of information and ensure compatibility. Broadcasting is
    right-aligned, (a, b, c, d) :: (c, d) is broadcast to (a, b, c, d), and
    (a, b, 1, d) :: (c, d) is broadcast to (a, b, c, d).

    Parameters
    ----------
    x1: Tensor
        First term, shape S.
    x2: Tensor
        Second term, shape T.
    out: Tensor | None = None
        Optional result container to avoid new copy.

    Returns
    -------
    Tensor
        New tensor containing the coordinate-wise subtraction.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("tensor", sp.BroadcastableTo(0), sp.Real()),
    )

    def forward(self, x1: Tensor, x2: Tensor, out: Tensor | None = None) -> Tensor:
        self._shapes = (x1.shape, x2.shape)
        return _binary_kernel_op(x1, x2, _C.subtract, _C.sub_inplace, out=out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        s1, s2 = self._shapes
        return (
            unbroadcast(grad_out, s1),
            unbroadcast(-1 * grad_out, s2),
        )

    @classmethod
    def flops(cls, x1: Tensor, x2: Tensor, _: Tensor | None = None) -> int:
        return math.prod(broadcast_shapes(x1.shape, x2.shape))

    @classmethod
    def mem_bytes(cls, x1: Tensor, x2: Tensor, _: Tensor | None = None) -> int:
        shape = broadcast_shapes(x1.shape, x2.shape)
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        return 3 * math.prod(shape) * sizeof(dtype)


class MulOp(Function):
    """Multiply two tensors component-wise with broadcast.

    Output tensor y is promoted and broadcast to the best suited dtype and shape
    to prevent loss of information and ensure compatibility. Broadcasting is
    right-aligned, (a, b, c, d) :: (c, d) is broadcast to (a, b, c, d), and
    (a, b, 1, d) :: (c, d) is broadcast to (a, b, c, d).

    Parameters
    ----------
    x1: Tensor
        First term, shape S.
    x2: Tensor
        Second term, shape T.
    out: Tensor | None = None
        Optional result container to avoid new copy.

    Returns
    -------
    Tensor
        New tensor containing the coordinate-wise multiplication.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("tensor", sp.BroadcastableTo(0), sp.Real()),
    )

    def forward(self, x1: Tensor, x2: Tensor, out: Tensor | None = None) -> Tensor:
        self.save_for_backward(x1, x2)
        return _binary_kernel_op(x1, x2, _C.multiply, _C.mul_inplace, out=out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x1, x2 = self.saved_tensors
        return (
            unbroadcast(x2 * grad_out, x1.shape),
            unbroadcast(x1 * grad_out, x2.shape),
        )

    @classmethod
    def flops(cls, x1: Tensor, x2: Tensor, _: Tensor | None = None) -> int:
        return math.prod(broadcast_shapes(x1.shape, x2.shape))

    @classmethod
    def mem_bytes(cls, x1: Tensor, x2: Tensor, _: Tensor | None = None) -> int:
        shape = broadcast_shapes(x1.shape, x2.shape)
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        return 3 * math.prod(shape) * sizeof(dtype)


class TrueDivOp(Function):
    """Floating point division with broadcast.

    Output tensor y is promoted and broadcast to the best suited dtype and shape
    to prevent loss of information and ensure compatibility. Broadcasting is
    right-aligned, (a, b, c, d) :: (c, d) is broadcast to (a, b, c, d), and
    (a, b, 1, d) :: (c, d) is broadcast to (a, b, c, d).

    Note: 1 / tensor(0.0) => tensor(inf) -- no error raised

    Parameters
    ----------
    x1: Tensor
        First term, shape S.
    x2: Tensor
        Second term, shape T.
    out: Tensor | None = None
        Optional result container to avoid new copy.

    Returns
    -------
    Tensor
        New tensor containing the coordinate-wise division.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("tensor", sp.BroadcastableTo(0), sp.Real()),
    )

    def forward(self, x1: Tensor, x2: Tensor, out: Tensor | None = None) -> Tensor:
        self.save_for_backward(x1, x2)
        return _binary_kernel_op(x1, x2, _C.divide, _C.div_inplace, out=out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x1, x2 = self.saved_tensors
        return (
            unbroadcast(grad_out / x2, x1.shape),
            unbroadcast(-x1 / x2**2 * grad_out, x2.shape),
        )

    @classmethod
    def flops(cls, x1: Tensor, x2: Tensor, _: Tensor | None = None) -> int:
        return math.prod(broadcast_shapes(x1.shape, x2.shape))

    @classmethod
    def mem_bytes(cls, x1: Tensor, x2: Tensor, _: Tensor | None = None) -> int:
        shape = broadcast_shapes(x1.shape, x2.shape)
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        return 3 * math.prod(shape) * sizeof(dtype)


class MatmulOp(Function):
    """Standard matrix multiplication.

    Available operations and shape broadcasting:
    - (k,) @ (k,) -> ()
    - (n, k) @ (k,) -> (n,)
    - (k,) @ (k, m) -> (m,)
    - (n, k) @ (k, m) -> (n, m)
    - (B1, n, k) @ (B2, k, m) -> (B1::B2, n, m) -- Broadcast B1 and B2

    Parameters
    ----------
    x1: Tensor
        First term, shape (a, b)
    x2: Tensor
        Second term, shape (b, c)

    Returns
    -------
    Tensor
        New tensor of shape (a, c) containing x1 @ x2.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(min_ndim=2), sp.Real()),
        sp.Input("tensor", sp.MatmulBroadcast(0), sp.Real()),
    )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        self.save_for_backward(x1, x2)
        self._shape1, self._shape2, shape_out = _matmul_broadcast(x1.shape, x2.shape)
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        x1 = x1.to(dtype).expand(*self._shape1)
        if x2.ndim == 1 and self._shape2[-2:] == (x2.shape[0], 1):
            x2 = x2.reshape(x2.shape[0], 1)  # Cannot expand to the right
        x2 = x2.to(dtype).expand(*self._shape2)
        return Tensor._new_contiguous(_C.matmul(x1._C_view, x2._C_view), shape_out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x1, x2 = self.saved_tensors
        x1_m = x1.expand(*self._shape1)
        if x2.ndim == 1 and self._shape2[-2:] == (x2.shape[0], 1):
            x2 = x2.reshape(x2.shape[0], 1)
        x2_m = x2.expand(*self._shape2)
        grad_out = grad_out.reshape(
            *broadcast_shapes(self._shape1[:-2], self._shape2[:-2]),
            self._shape1[-2],
            self._shape2[-1],
        )
        return (
            unbroadcast(grad_out @ x2_m.transpose(-2, -1), x1.shape).reshape(*x1.shape),
            unbroadcast(x1_m.transpose(-2, -1) @ grad_out, x2.shape).reshape(*x2.shape),
        )

    @classmethod
    def flops(cls, x1: Tensor, x2: Tensor) -> int:
        shape1, shape2, _ = _matmul_broadcast(x1.shape, x2.shape)
        B, M, N, K = math.prod(shape1[:-2]), shape1[-2], shape2[-1], shape1[-1]
        return 2 * B * M * N * K

    @classmethod
    def mem_bytes(cls, x1: Tensor, x2: Tensor) -> int:
        shape1, shape2, _ = _matmul_broadcast(x1.shape, x2.shape)
        B, M, N, K = math.prod(shape1[:-2]), shape1[-2], shape2[-1], shape1[-1]
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        return B * (M * K + K * N + M * N) * sizeof(dtype)


def _binary_kernel_op(
    x1: Tensor,
    x2: Tensor,
    op: Callable[[_C.TensorView, _C.TensorView], _C.Storage],
    op_inplace: Callable[[_C.TensorView, _C.TensorView], None],
    out: Tensor | None = None,
) -> Tensor:
    """Shortcut for any binary operator kernel."""
    out_set = out is not None
    if out_set:
        if out.is_leaf and out.requires_grad and is_grad_enabled():
            raise RuntimeError(
                "A leaf variable that requires grad is being used in an in-place op."
            )
        dtype = out.dtype
        shape = out.shape
    else:
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        shape = broadcast_shapes(x1.shape, x2.shape)

    out_is_x1 = out is x1  # save before rebind
    x1 = x1.to(dtype).expand(*shape)
    x2 = x2.to(dtype).expand(*shape)

    if out is None:
        return Tensor._new_contiguous(op(x1._C_view, x2._C_view), shape)
    if out_is_x1:
        op_inplace(x1._C_view, x2._C_view)
        return x1

    result = Tensor._new_contiguous(op(x1._C_view, x2._C_view), shape)
    _C.copy_inplace(out._C_view, result._C_view)
    return result


def _matmul_broadcast(
    x1: TensorShape, x2: TensorShape
) -> tuple[TensorShape, TensorShape, TensorShape]:
    """Carries out matrix multiplication shape broadcast."""
    if len(x1) == 0 or len(x2) == 0:
        raise RuntimeError("Scalar value encountered in matmul.")
    dim_to_remove = []
    if len(x1) == 1:
        x1 = (1, *x1)
        dim_to_remove.append(-2)
    if len(x2) == 1:
        x2 = (*x2, 1)
        dim_to_remove.append(-1)
    if x1[-1] != x2[-2]:
        raise RuntimeError(f"Cannot multiply {x1} @ {x2}")
    bshape = broadcast_shapes(x1[:-2], x2[:-2])
    shape_out_raw = list(bshape) + [x1[-2], x2[-1]]
    dim_to_remove = [ax if ax >= 0 else ax + len(shape_out_raw) for ax in dim_to_remove]
    shape_out = [
        dim_i for i, dim_i in enumerate(shape_out_raw) if i not in dim_to_remove
    ]
    return (
        tuple(list(bshape) + list(x1[-2:])),
        tuple(list(bshape) + list(x2[-2:])),
        tuple(shape_out),
    )
