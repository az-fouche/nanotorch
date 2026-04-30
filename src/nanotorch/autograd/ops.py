"""Usual differentiable tensor operations and their derivative."""

from typing import Callable

from nanotorch import _C
from nanotorch._data_type import DataType, promote_dtypes
from nanotorch._indexing import TensorShape, broadcast_shapes
from nanotorch.core import Tensor, TensorLike

from .function import Function


class AddOp(Function):
    """y = x1 + x2 operation."""

    def forward(self, x1: Tensor, x2: TensorLike) -> Tensor:
        self._shapes = (x1.shape, x2.shape if isinstance(x2, Tensor) else None)
        return _binary_kernel_op(x1, x2, _C.add)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return tuple(
            _unbroadcast(grad_out, shape) for shape in self._shapes if shape is not None
        )


class MulOp(Function):
    """y = x1 * x2 operation."""

    def forward(self, x1: Tensor, x2: TensorLike) -> Tensor:
        if isinstance(x2, Tensor):
            self.save_for_backward(x1, x2)
            self._x2 = None
        else:
            self.save_for_backward(x1)
            self._x2 = x2
        return _binary_kernel_op(x1, x2, _C.multiply)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        match len(self.saved_tensors):
            case 1:
                if self._x2 is None:
                    raise RuntimeError("x2 was not saved.")
                return (self._x2 * grad_out,)
            case 2:
                x1, x2 = self.saved_tensors
                return (
                    _unbroadcast(x2 * grad_out, x1.shape),
                    _unbroadcast(x1 * grad_out, x2.shape),
                )
            case _:
                raise RuntimeError("Multiply function only supports 1 or 2 tensors.")


class SubOp(Function):
    """y = x1 - x2 operation."""

    def forward(self, x1: Tensor, x2: TensorLike) -> Tensor:
        self._shapes = (x1.shape, x2.shape if isinstance(x2, Tensor) else None)
        return _binary_kernel_op(x1, x2, _C.subtract)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        s1, s2 = self._shapes
        if s2 is None:
            return (_unbroadcast(grad_out, s1),)
        return (
            _unbroadcast(grad_out, s1),
            _unbroadcast(-1 * grad_out, s2),
        )


class ExpOp(Function):
    """y = e^x operation."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, DataType.FP32)
        x = x.to(dtype)
        e_x = Tensor._new_contiguous(x.dtype, x.shape, _C.exp(x._C_view))
        self.save_for_backward(e_x)
        return e_x

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if len(self.saved_tensors) > 1:
            raise RuntimeError("Exp function only supports 1 tensor.")
        e_x: Tensor = self.saved_tensors[0]  # type: ignore
        return (grad_out * e_x,)


class PowOp(Function):
    """y = x^a operation."""

    def forward(self, x: Tensor, exponent: Tensor | float | bool | int) -> Tensor:
        if x.is_empty:
            return x
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)
        dtype = promote_dtypes(x.dtype, exponent.dtype)
        exponent_fp = float(exponent.item())
        if exponent_fp < 0 and dtype <= DataType.INT64:
            raise RuntimeError("Cannot use negative exponent with integer tensor.")
        self.save_for_backward(x, exponent)
        return Tensor._new_contiguous(
            dtype, x.shape, _C.pow(x.to(dtype)._C_view, exponent_fp)
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x, exponent = self.saved_tensors  # type: ignore
        return (grad_out * exponent * x ** (exponent - 1),)


class LogOp(Function):
    """y = log(x) operation."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, DataType.FP32)
        x = x.to(dtype)
        self.save_for_backward(x)
        return Tensor._new_contiguous(dtype, x.shape, _C.log(x._C_view))

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if len(self.saved_tensors) > 1:
            raise RuntimeError("Log function only supports 1 tensor.")
        x: Tensor = self.saved_tensors[0]  # type: ignore
        return (grad_out * x**-1,)


class SumOp(Function):
    """y = sum(x, axis=...) operation."""

    def forward(
        self,
        x: Tensor,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        dtype: DataType | None = None,
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
            dtype = max(x.dtype, DataType.INT64)

        if x.is_empty:
            return Tensor._new_contiguous(
                dtype, new_shape, _C.zeros(new_shape, dtype.cpp_dtype)
            )
        self._orig_shape = x.shape

        return Tensor._new_contiguous(
            dtype=dtype,
            shape=new_shape,
            storage=_C.sum(x._C_view, axis, keepdim, dtype.cpp_dtype),
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if self._orig_shape is None:
            return (grad_out,)
        return (grad_out.expand(self._orig_shape),)


class MeanOp(Function):
    """y = mean(x, axis=...) operation."""

    def forward(
        self,
        x: Tensor,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        dtype: DataType | None = None,
    ):
        if dtype is None:
            dtype = x.dtype
        if dtype < DataType.FP32:
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
            x.dtype,
            tuple(reversed(x.shape)),
            x.storage,
            tuple(reversed(strides)),
            offset,
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out.T,)


class MatmulOp(Function):
    """y = x1 @ x2 operation."""

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError("Only 2D matmul is supported.")
        if x1.shape[1] != x2.shape[0]:
            raise ValueError(f"Cannot multiply {x1.shape} @ {x2.shape}.")
        self.save_for_backward(x1, x2)
        shape = (x1.shape[0], x2.shape[1])
        dtype = promote_dtypes(x1.dtype, x2.dtype)
        x1 = x1.to(dtype)
        x2 = x2.to(dtype)
        return Tensor._new_contiguous(
            dtype=dtype, shape=shape, storage=_C.matmul(x1._C_view, x2._C_view)
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x1, x2 = self.saved_tensors
        return (grad_out @ x2.T, x1.T @ grad_out)


class ReluOp(Function):
    """Rectified linear unit."""

    def forward(self, x: Tensor) -> Tensor:
        result = Tensor._new_contiguous(x.dtype, x.shape, _C.relu(x._C_view))
        self.save_for_backward(result)
        return result

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        result = self.saved_tensors[0]
        return (
            grad_out
            * Tensor._new_contiguous(
                result.dtype, result.shape, _C.greater(result._C_view, 0)
            ),
        )


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
    x1: TensorLike,
    x2: TensorLike,
    op: Callable[[_C.TensorView, _C.TensorView], _C.Storage],
) -> Tensor:
    """Shortcut for any binary operator kernel."""
    if not isinstance(x1, Tensor):
        x1 = Tensor(x1)
    if not isinstance(x2, Tensor):
        x2 = Tensor(x2)
    dtype = promote_dtypes(x1.dtype, x2.dtype)
    shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = x1.to(dtype).expand(shape)
    x2 = x2.to(dtype).expand(shape)
    return Tensor._new_contiguous(
        dtype=dtype, shape=shape, storage=op(x1._C_view, x2._C_view)
    )
