"""Common helpers for tensor ops."""

from nanotorch._indexing import TensorShape
from nanotorch.core import Tensor


def unbroadcast(x: Tensor, shape: TensorShape) -> Tensor:
    """Unbroadcast by summing (2, 3, 4, 4) -> (4, 4)"""
    leading = tuple(range(len(x.shape) - len(shape)))
    if leading:
        x = x.sum(axis=leading)
    trailing = tuple(
        ax for ax in range(len(shape)) if x.shape[ax] != 1 and shape[ax] == 1
    )
    if trailing:
        x = x.sum(axis=trailing, keepdim=True)
    return x


def broadcast_axis(axis: int | TensorShape | None, x: Tensor) -> TensorShape:
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None or len(axis) == 0:
        axis = tuple(range(x.ndim))
    axis = tuple(ax + x.ndim if ax < 0 else ax for ax in axis)
    if any(not isinstance(ax, int) for ax in axis):
        raise TypeError(f"Invalid axis: {axis}")
    if len(set(axis)) != len(axis):
        raise ValueError(f"Redundant axes: {axis}")
    if any(ax < 0 or ax >= x.ndim for ax in axis):
        raise IndexError(f"Invalid axis: {axis}")
    return axis
