"""Usual differentiable tensor operations and their derivative."""

import math

from nanotorch import _C
from nanotorch._data_type import Dtype
from nanotorch._indexing import TensorShape
from nanotorch.core import Tensor

from . import ops_spec as sp
from .function import Function
from .ops_utils import broadcast_axis, unbroadcast


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

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real()),)

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

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real()),)

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


class ReshapeOp(Function):
    """Returns a reshaped view of the tensor (does not copy).

    Parameters
    ----------
    x: Tensor
        Tensor to reshape.
    dims: *int
        New shape dimensions, prod(dims) should equal tensor's number of
        elements, otherwise reshape raises a ValueError.

    Returns
    -------
    Tensor
        Reshaped tensor view.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("scalar", None, sp.AxisReshape(0)),
    )

    def forward(self, x: Tensor, *dims: int):
        if math.prod(dims) != x.numel:
            raise ValueError(f"Can't reshape {x.shape} into {dims}.")
        self._shape = x.shape
        if dims == x.shape:
            return x
        x = x._to_contiguous()
        return Tensor._new_view(
            x.storage, shape=tuple(dims), strides=None, offset=x._offset
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out.reshape(*self._shape),)


class ExpandOp(Function):
    """Expand the tensor to target shape, no copy.

    Parameters
    ----------
    x: Tensor
        Tensor to expand.
    dims: *int
        Shape to expand the tensor to, last dimensions must match the tensor
        dimension. If tensor dimension self.shape[i] is 1, it is broadcasted to
        shape[i].
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real()),
        sp.Input("scalar", None, sp.AxisExpand(0)),
    )

    def forward(self, x: Tensor, *dims: int) -> Tensor:
        if len(dims) < x.ndim:
            raise ValueError(f"Cannot broadcast {x.shape} to {dims}.")

        self._orig_shape = x.shape
        if dims == x.shape:
            return x

        strides = list(x._strides)
        for i in range(len(dims)):
            if i >= x.ndim:
                strides = [0] + strides
            else:
                src, tgt = x.shape[-i - 1], dims[-i - 1]
                if src == tgt:
                    continue
                elif src == 1:
                    strides[-i - 1] = 0  # broadcast
                else:
                    raise ValueError(
                        f"Cannot broadcast {x.shape} to {dims} (mismatch)."
                    )

        return Tensor._new_view(x.storage, dims, tuple(strides), x._offset)

    def backward(self, grad_out: Tensor) -> tuple[Tensor]:
        return (unbroadcast(grad_out, self._orig_shape),)


class TOp(Function):
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

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real()),)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out.T,)


class TransposeOp(Function):
    """Permutes two tensor dimensions (does not copy).

    Parameters
    ----------
    x: Tensor
        Tensor to transpose.
    dim0: int
        Index of the first dimension to permute.
    dim1:
        Index of the second dimension to permute.

    Returns
    -------
        Tensor view with swapped dimensions.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(min_ndim=2), sp.Real()),
        sp.Input("scalar", None, sp.Axis(0)),
        sp.Input("scalar", None, sp.Axis(0)),
    )

    def forward(self, x: Tensor, dim0: int, dim1: int) -> Tensor:
        self._dim0 = dim0
        self._dim1 = dim1
        if len(x.shape) < 2:
            return x
        if dim0 == dim1:
            return x
        tstrides, offset = x.strides
        shape, strides = list(x.shape), list(tstrides)
        shape[dim0], shape[dim1], strides[dim0], strides[dim1] = (
            shape[dim1],
            shape[dim0],
            strides[dim1],
            strides[dim0],
        )
        return Tensor._new_view(
            x.storage,
            shape=tuple(shape),
            strides=tuple(strides),
            offset=offset,
        )

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out.transpose(self._dim0, self._dim1),)
