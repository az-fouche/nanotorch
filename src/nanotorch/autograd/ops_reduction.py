"""Reduction differentiable tensor operations and their derivative."""

import math

from nanotorch import _C  # type: ignore[missing-import]
from nanotorch._data_type import Dtype
from nanotorch._indexing import TensorShape
from nanotorch.core import Tensor

from . import ops_spec as sp
from .function import Function
from .ops_utils import broadcast_axis


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

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("scalar", None, sp.AxisSet(ref=0, min=0, split=False)),
        sp.Input("scalar", None, sp.Bool()),
    )

    def forward(
        self,
        x: Tensor,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        dtype: Dtype | None = None,
    ) -> Tensor:
        self._axis = broadcast_axis(axis, x)

        if dtype is None:
            dtype = max(x.dtype, Dtype.Int64)
        if keepdim:
            new_shape = tuple(
                1 if i in self._axis else s for i, s in enumerate(x.shape)
            )
        else:
            new_shape = tuple(s for i, s in enumerate(x.shape) if i not in self._axis)

        if x.is_empty:
            return Tensor._new_contiguous(
                _C.zeros(math.prod(new_shape), dtype, x.device), new_shape
            )
        self._orig_shape = x.shape
        return Tensor._new_contiguous(_C.sum(x._C_view, self._axis, dtype), new_shape)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if not self._orig_shape:
            return (grad_out,)
        old_shape_dim = tuple(
            1 if dim in self._axis else self._orig_shape[dim]
            for dim in range(len(self._orig_shape))
        )
        return (grad_out.reshape(*old_shape_dim).expand(*self._orig_shape),)


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

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("scalar", None, sp.AxisSet(ref=0, min=0, split=False)),
        sp.Input("scalar", None, sp.Bool()),
    )

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
        self._axis = broadcast_axis(axis, x)
        sum_ = x.sum(axis=self._axis, keepdim=keepdim, dtype=dtype)
        self._orig_shape = x.shape
        self._denom = x.numel // sum_.numel
        return sum_ / self._denom

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if not self._orig_shape:
            return (grad_out,)
        old_shape_dim = tuple(
            1 if dim in self._axis else self._orig_shape[dim]
            for dim in range(len(self._orig_shape))
        )
        return (
            grad_out.reshape(*old_shape_dim).expand(*self._orig_shape) / self._denom,
        )


class MaxOp(Function):
    """Computes the max along one or more axes.

    Parameters
    ----------
    x: Tensor
        Target tensor.
    axis: int | tuple[int, ...] | None
        Axes index to max along, integer is interpreted as (n,), None as all axes.
        Dimensions of these axes are collapsed by the operation.
    keepdim: bool
        Instead of collapsing target axes, keeps the axis with shape[i] = 1.

    Returns
    -------
    Tensor
        New tensor of containing the max coefficients.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("scalar", None, sp.AxisSet(ref=0, min=0, split=False)),
        sp.Input("scalar", None, sp.Bool()),
    )

    def forward(
        self,
        x: Tensor,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        self._axis = broadcast_axis(axis, x)
        if keepdim:
            new_shape = tuple(
                1 if i in self._axis else s for i, s in enumerate(x.shape)
            )
        else:
            new_shape = tuple(s for i, s in enumerate(x.shape) if i not in self._axis)

        if x.is_empty:
            if math.prod(new_shape) != 0:
                raise IndexError(
                    "argmin() expects reduction dim to be specified for numel=0."
                )
            return Tensor._new_contiguous(
                _C.zeros(math.prod(new_shape), x.dtype, x.device), new_shape
            )
        self._orig_shape = x.shape
        max_ = Tensor._new_contiguous(_C.max(x._C_view, self._axis), new_shape)
        self.save_for_backward(x, max_)
        return max_

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if not self._orig_shape:
            return (grad_out,)
        old_shape_dim = tuple(
            1 if dim in self._axis else self._orig_shape[dim]
            for dim in range(len(self._orig_shape))
        )
        x, max_ = self.saved_tensors
        # Flow 1/kmax gradient over every max occurence
        mask = x == max_.reshape(*old_shape_dim).expand(*self._orig_shape)
        mask *= 1 / mask.sum(axis=self._axis, keepdim=True)
        return (grad_out.reshape(*old_shape_dim).expand(*self._orig_shape) * mask,)


class MinOp(Function):
    """Computes the min along one or more axes.

    Parameters
    ----------
    x: Tensor
        Target tensor.
    axis: int | tuple[int, ...] | None
        Axes index to min along, integer is interpreted as (n,), None as all axes.
        Dimensions of these axes are collapsed by the operation.
    keepdim: bool
        Instead of collapsing target axes, keeps the axis with shape[i] = 1.

    Returns
    -------
    Tensor
        New tensor of containing the min coefficients.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("scalar", None, sp.AxisSet(ref=0, min=0, split=False)),
        sp.Input("scalar", None, sp.Bool()),
    )

    def forward(
        self,
        x: Tensor,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        self._axis = broadcast_axis(axis, x)
        if keepdim:
            new_shape = tuple(
                1 if i in self._axis else s for i, s in enumerate(x.shape)
            )
        else:
            new_shape = tuple(s for i, s in enumerate(x.shape) if i not in self._axis)

        if x.is_empty:
            if math.prod(new_shape) != 0:
                raise IndexError(
                    "argmin() expects reduction dim to be specified for numel=0."
                )
            return Tensor._new_contiguous(
                _C.zeros(math.prod(new_shape), x.dtype, x.device), new_shape
            )
        self._orig_shape = x.shape
        min_ = Tensor._new_contiguous(_C.min(x._C_view, self._axis), new_shape)
        self.save_for_backward(x, min_)
        return min_

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        if not self._orig_shape:
            return (grad_out,)
        old_shape_dim = tuple(
            1 if dim in self._axis else self._orig_shape[dim]
            for dim in range(len(self._orig_shape))
        )
        x, min_ = self.saved_tensors
        # Flow 1/kmin gradient over every min occurence
        mask = x == min_.reshape(*old_shape_dim).expand(*self._orig_shape)
        mask *= 1 / mask.sum(axis=self._axis, keepdim=True)
        return (grad_out.reshape(*old_shape_dim).expand(*self._orig_shape) * mask,)
