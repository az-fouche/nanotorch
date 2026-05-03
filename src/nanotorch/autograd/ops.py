"""Usual differentiable tensor operations and their derivative."""

import math
from typing import Callable

from nanotorch import _C
from nanotorch._data_type import Dtype, promote_dtypes
from nanotorch._indexing import TensorShape, broadcast_shapes
from nanotorch.core import Tensor

from .function import Function
from .grad_mode import is_grad_enabled


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

    def forward(self, x1: Tensor, x2: Tensor, out: Tensor | None = None) -> Tensor:
        self._shapes = (x1.shape, x2.shape)
        return _binary_kernel_op(x1, x2, _C.add, _C.add_inplace, out=out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return tuple(_unbroadcast(grad_out, shape) for shape in self._shapes)


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

    def forward(self, x1: Tensor, x2: Tensor, out: Tensor | None = None) -> Tensor:
        self._shapes = (x1.shape, x2.shape)
        return _binary_kernel_op(x1, x2, _C.subtract, _C.sub_inplace, out=out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        s1, s2 = self._shapes
        return (
            _unbroadcast(grad_out, s1),
            _unbroadcast(-1 * grad_out, s2),
        )


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

    def forward(self, x1: Tensor, x2: Tensor, out: Tensor | None = None) -> Tensor:
        self.save_for_backward(x1, x2)
        return _binary_kernel_op(x1, x2, _C.multiply, _C.mul_inplace, out=out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x1, x2 = self.saved_tensors
        return (
            _unbroadcast(x2 * grad_out, x1.shape),
            _unbroadcast(x1 * grad_out, x2.shape),
        )


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

    def forward(self, x1: Tensor, x2: Tensor, out: Tensor | None = None) -> Tensor:
        self.save_for_backward(x1, x2)
        return _binary_kernel_op(x1, x2, _C.divide, _C.div_inplace, out=out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x1, x2 = self.saved_tensors
        return (
            _unbroadcast(grad_out / x2, x1.shape),
            _unbroadcast(-x1 / x2**2 * grad_out, x2.shape),
        )


class NegOp(Function):
    """Sign inversion operation.

    Parameters
    ----------
    x1: Tensor
        Tensor to invert coefficients (raises if bool).

    Returns
    -------
    Tensor
        New tensor containing the coordinate-wise division.
    """

    def forward(self, x1: Tensor) -> Tensor:
        self._shape = x1.shape
        return Tensor._new_contiguous(_C.neg(x1._C_view), x1.shape)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (_unbroadcast(-1 * grad_out, self._shape),)


class ExpOp(Function):
    """Exponentiate all tensor coefficients.

    Parameters
    ----------
    x: Tensor
        Target tensor to exponentiate.

    Returns
    -------
    Tensor
        Exponentiated tensor, cast to floating point if necessary.
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, Dtype.Float32)
        x = x.to(dtype)
        e_x = Tensor._new_contiguous(_C.exp(x._C_view), x.shape)
        self.save_for_backward(e_x)
        return e_x

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out * self.saved_tensors[0],)


class LogOp(Function):
    """Logarithmize (Napierian) all tensor coefficients.

    Parameters
    ----------
    x: Tensor
        Target tensor to logarithmize.

    Returns
    -------
    Tensor
        Logarithmized tensor, cast to floating point if necessary.
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, Dtype.Float32)
        x = x.to(dtype)
        self.save_for_backward(x)
        return Tensor._new_contiguous(_C.log(x._C_view), x.shape)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out * self.saved_tensors[0] ** -1,)


class PowOp(Function):
    """Raises all tensor coefficients to a given exponent.

    Parameters
    ----------
    x: Tensor
        Target tensor.
    exponent: Tensor | float | int | bool
        Exponent, must be scalar-like.

    Returns
    -------
    Tensor
        Resulting tensor.
    """

    def forward(self, x: Tensor, exponent: Tensor) -> Tensor:
        if x.is_empty:
            return x
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)
        dtype = promote_dtypes(x.dtype, exponent.dtype)
        exponent_fp = float(exponent.item())
        if exponent_fp < 0 and dtype <= Dtype.Int64:
            raise RuntimeError("Cannot use negative exponent with integer tensor.")
        self.save_for_backward(x, exponent)
        return Tensor._new_contiguous(_C.pow(x.to(dtype)._C_view, exponent_fp), x.shape)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x, exponent = self.saved_tensors
        return (grad_out * exponent * x ** (exponent - 1),)


class SumOp(Function):
    """Sum the tensor elements along one or more axes.

    Parameters
    ----------
    x: Tensor
        Target tensor.
    axis: int | tuple[int, ...] | None
        Axes index to sum along, integer is interpreted as (n,), None as all axes.
        Dimensions of these axes are collapsed by the operation.
    keepdim: bool
        Instead of collapsing target axes, keeps the axis with shape[i] = 1.
    dtype: Dtype | None
        Casts the resulting sum to this data type.

    Returns
    -------
    Tensor
        New tensor of containing the summed coefficients.
    """

    def forward(
        self,
        x: Tensor,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        dtype: Dtype | None = None,
    ) -> Tensor:
        if axis is None:
            axis = tuple(range(x.ndim))

        self._axis = axis
        self._keepdim = keepdim

        if axis == ():
            self._orig_shape = None
            return x.to(dtype or x.dtype)
        elif isinstance(axis, int):
            axis = (axis,)

        axis = tuple(ax + x.ndim if ax < 0 else ax for ax in axis)
        if any(not isinstance(ax, int) for ax in axis):
            raise TypeError(f"Invalid axis: {axis}")
        if len(set(axis)) != len(axis):
            raise ValueError(f"Redundant axes: {axis}")
        if any(ax < 0 or ax >= x.ndim for ax in axis):
            raise IndexError(f"Invalid axis: {axis}")
        if keepdim:
            new_shape = tuple(1 if i in axis else s for i, s in enumerate(x.shape))
        else:
            new_shape = tuple(s for i, s in enumerate(x.shape) if i not in axis)

        if dtype is None:
            dtype = max(x.dtype, Dtype.Int64)

        if x.is_empty:
            return Tensor._new_contiguous(
                _C.zeros(math.prod(new_shape), dtype, x.device), new_shape
            )
        self._orig_shape = x.shape

        return Tensor._new_contiguous(
            _C.sum(x._C_view, axis, keepdim, dtype), new_shape
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if self._orig_shape is None:
            return (grad_out,)
        return (grad_out.expand(self._orig_shape),)


class MeanOp(Function):
    """Averages the tensor elements along one or more axes.

    Parameters
    ----------
    x: Tensor
        Target tensor.
    axis: int | tuple[int, ...] | None
        Axes index to average along, integer is interpreted as (n,), None as all axes.
        Dimensions of these axes are collapsed by the operation.
    keepdim: bool
        Instead of collapsing target axes, keeps the axis with shape[i] = 1.
    dtype: Dtype | None
        Casts the resulting mean to this data type.

    Returns
    -------
    Tensor
        New tensor of containing the averaged coefficients.
    """

    def forward(
        self,
        x: Tensor,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        dtype: Dtype | None = None,
    ):
        if dtype is None:
            dtype = x.dtype
        if dtype < Dtype.Float32:
            raise TypeError("Cannot take mean of integer tensor, cast to fp.")
        sum_ = x.sum(axis=axis, keepdim=keepdim, dtype=dtype)
        self._orig_shape = x.shape
        self._denom = x.numel // sum_.numel
        return sum_ / self._denom

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if self._orig_shape is None:
            return (grad_out / self._denom,)
        return (grad_out.expand(self._orig_shape) / self._denom,)


class TransposeOp(Function):
    """y = x.T operation."""

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) < 2:
            return x
        strides, offset = x.strides
        return Tensor._new_view(
            x.storage,
            tuple(reversed(x.shape)),
            tuple(reversed(strides)),
            offset,
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out.T,)


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

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        self.save_for_backward(x1, x2)
        self._shape1, self._shape2, shape_out = _matmul_broadcast(x1.shape, x2.shape)
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        x1 = x1.to(dtype).expand(self._shape1)
        if x2.ndim == 1 and self._shape2[-2:] == (x2.shape[0], 1):
            x2 = x2.reshape(x2.shape[0], 1)  # Cannot expand to the right
        x2 = x2.to(dtype).expand(self._shape2)
        return Tensor._new_contiguous(_C.matmul(x1._C_view, x2._C_view), shape_out)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x1, x2 = self.saved_tensors
        x1_m = x1.expand(self._shape1)
        if x2.ndim == 1 and self._shape2[-2:] == (x2.shape[0], 1):
            x2 = x2.reshape(x2.shape[0], 1)
        x2_m = x2.expand(self._shape2)
        grad_out = grad_out.reshape(
            *broadcast_shapes(self._shape1[:-2], self._shape2[:-2]),
            self._shape1[-2],
            self._shape2[-1],
        )
        return (
            _unbroadcast(grad_out @ x2_m.transpose(-2, -1), x1.shape).reshape(
                *x1.shape
            ),
            _unbroadcast(x1_m.transpose(-2, -1) @ grad_out, x2.shape).reshape(
                *x2.shape
            ),
        )


class ReluOp(Function):
    """Rectified linear unit."""

    def forward(self, x: Tensor) -> Tensor:
        result = Tensor._new_contiguous(_C.relu(x._C_view), x.shape)
        self.save_for_backward(result)
        return result

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        result = self.saved_tensors[0]
        return (grad_out * greater_op(result, Tensor(0)),)


def equal_op(x1: Tensor, x2: Tensor) -> Tensor:
    dtype = promote_dtypes(x1.dtype, x2.dtype)
    shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = x1.to(dtype).expand(shape)
    x2 = x2.to(dtype).expand(shape)
    return Tensor._new_contiguous(_C.pw_equal(x1._C_view, x2._C_view), x1.shape)


def greater_op(x1: Tensor, x2: Tensor) -> Tensor:
    dtype = promote_dtypes(x1.dtype, x2.dtype)
    shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = x1.to(dtype).expand(shape)
    x2 = x2.to(dtype).expand(shape)
    return Tensor._new_contiguous(_C.pw_greater(x1._C_view, x2._C_view), x1.shape)


def greater_eq_op(x1: Tensor, x2: Tensor) -> Tensor:
    dtype = promote_dtypes(x1.dtype, x2.dtype)
    shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = x1.to(dtype).expand(shape)
    x2 = x2.to(dtype).expand(shape)
    return Tensor._new_contiguous(_C.pw_greater_eq(x1._C_view, x2._C_view), x1.shape)


def _unbroadcast(x: Tensor, shape: TensorShape) -> Tensor:
    """Unbroadcast by summing (2, 3, 4, 4) -> (4, 4)"""
    x = x.sum(axis=tuple(range(len(x.shape) - len(shape))))
    return x.sum(
        axis=tuple(
            ax for ax in range(len(shape)) if x.shape[ax] != 1 and shape[ax] == 1
        ),
        keepdim=True,
    )


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
    x1 = x1.to(dtype).expand(shape)
    x2 = x2.to(dtype).expand(shape)

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
