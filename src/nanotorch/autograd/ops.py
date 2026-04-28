"""Usual differentiable tensor operations and their derivative."""

from typing import Callable

from nanotorch import _C
from nanotorch._data_type import DataType, promote_dtypes
from nanotorch._indexing import TensorShape, broadcast_shapes
from nanotorch.core import Tensor, TensorLike

from .function import Function


class AddOp(Function):
    """y = x1 + x2 functor."""

    def forward(self, x1: Tensor, x2: TensorLike) -> Tensor:
        if isinstance(x2, Tensor):
            self.save_for_backward(x1, x2)
        else:
            self.save_for_backward(x1)
        return _binary_kernel_op(x1, x2, _C.add)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if self.saved_tensors is None:
            raise RuntimeError("Cannot backward with no saved tensors.")
        if len(self.saved_tensors) > 2:
            raise RuntimeError("Add function only supports 1 or 2 tensors.")
        out = []
        for x in self.saved_tensors:
            if not isinstance(x, Tensor):
                raise RuntimeError(f"Cannot backward through x of type {type(x)}.")
            out.append(_unbroadcast(grad_out, x.shape))
        return tuple(out)


class MulOp(Function):
    """y = x1 * x2 functor."""

    def forward(self, x1: Tensor, x2: TensorLike) -> Tensor:
        self.save_for_backward(x1, x2)
        return _binary_kernel_op(x1, x2, _C.multiply)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if self.saved_tensors is None:
            raise RuntimeError("Cannot backward with no saved tensors.")
        if len(self.saved_tensors) > 2:
            raise RuntimeError("Multiply function only supports 1 or 2 tensors.")
        x1, x2 = self.saved_tensors  # type: ignore
        if not isinstance(x1, Tensor):
            raise RuntimeError(f"Cannot backward through x1 of type {type(x1)}.")
        if not isinstance(x2, Tensor):
            return (_unbroadcast(x2 * grad_out, x1.shape),)
        return (
            _unbroadcast(x2 * grad_out, x1.shape),
            _unbroadcast(x1 * grad_out, x2.shape),
        )


class ExpOp(Function):
    """y = e^x functor."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, DataType.FP32)
        x = x.to(dtype)
        self.save_for_backward(x)
        return Tensor._new_contiguous(x.dtype, x.shape, _C.exp(x._C_view))

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if self.saved_tensors is None:
            raise RuntimeError("Cannot backward with no saved tensors.")
        if len(self.saved_tensors) > 1:
            raise RuntimeError("Exp function only supports 1 tensor.")
        x: Tensor = self.saved_tensors[0]  # type: ignore
        e_x = Tensor._new_contiguous(x.dtype, x.shape, _C.exp(x._C_view))
        return (grad_out * e_x,)


class LogOp(Function):
    """y = log(x) functor."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, DataType.FP32)
        x = x.to(dtype)
        self.save_for_backward(x)
        return Tensor._new_contiguous(dtype, x.shape, _C.log(x._C_view))

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if self.saved_tensors is None:
            raise RuntimeError("Cannot backward with no saved tensors.")
        if len(self.saved_tensors) > 1:
            raise RuntimeError("Log function only supports 1 tensor.")
        x: Tensor = self.saved_tensors[0]  # type: ignore
        return (grad_out * x**-1,)


class SumOp(Function):
    """y = sum(x, axis=...) functor."""

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

        self.save_for_backward(x)

        return Tensor._new_contiguous(
            dtype=dtype,
            shape=new_shape,
            storage=_C.sum(x._C_view, axis, keepdim, dtype.cpp_dtype),
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if self.saved_tensors is None:  # Was called with ()
            if not hasattr(self, "_axis") or self._axis != ():
                raise ValueError("No tensor saved.")
            return (grad_out,)
        if len(self.saved_tensors) > 1:
            raise RuntimeError("Sum function only supports 1 tensor.")
        x: Tensor = self.saved_tensors[0]  # type: ignore
        return (grad_out.expand(x.shape),)


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
