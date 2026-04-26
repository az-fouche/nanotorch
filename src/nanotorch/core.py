"""Base tensor class."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, Callable, Sequence

from nanotorch import _C

from ._data_type import DataType, InputType, promote_dtypes
from ._indexing import (
    TensorShape,
    broadcast_shapes,
    expand_ellipsis,
    is_contiguous_view,
)

MAX_TENSOR_LEVEL = 32


class Tensor:
    """Base tensor class.

    Parameters
    ----------
    data: TensorInputType
        Tensor coefficients (scalar or list-like).
    dtype: DataType
        Data type (optional, otherwise will be autodetected).
    """

    def __init__(self, data: InputType, dtype: DataType | None = None):

        dtype, shape, storage = _extract_tensor_data(data, dtype)

        self._dtype = dtype
        self._shape = shape
        self._data = storage
        self._strides = _infer_strides(shape)
        self._offset = 0

    @classmethod
    def _new_contiguous(
        cls, dtype: DataType, shape: TensorShape, storage: _C.Storage
    ) -> Tensor:
        """Intializes a new contiguous tensor from precomputed components."""
        return cls._new_view(dtype, shape, storage, None, None)

    @classmethod
    def _new_view(
        cls,
        dtype: DataType,
        shape: TensorShape,
        storage: _C.Storage,
        strides: TensorShape | None,
        offset: int | None,
    ) -> Tensor:
        """Intializes a new tensor from precomputed components."""

        strides = _infer_strides(shape) if strides is None else strides
        offset = 0 if offset is None else offset
        min_index = offset + sum(
            (s - 1) * st for s, st in zip(shape, strides) if st < 0
        )
        max_index = offset + sum(
            (s - 1) * st for s, st in zip(shape, strides) if st > 0
        )
        max_storage = len(memoryview(storage))
        if math.prod(shape) > 0 and (min_index < 0 or max_index >= max_storage):
            raise IndexError(
                f"View max index {max_index} exceeds storage capacity {max_storage}."
            )

        new = cls.__new__(cls)
        new._dtype = dtype
        new._shape = shape
        new._data = storage
        new._strides = strides
        new._offset = offset
        return new

    def __repr__(self) -> str:
        if self.ndim == 0:
            return f"nt.Tensor({self.tolist()})"
        elif self.ndim == 1:
            return f"nt.Tensor({_str_list_compact(self.tolist())})"
        elif self.ndim == 2:
            if len(self) <= 5:
                indices = list(range(len(self)))
            else:
                indices = [0, 1, None, -2, -1]
            buff = "nt.Tensor("
            for ni, i in enumerate(indices):
                if ni > 0:
                    buff += "          "
                if i is None:
                    buff += "..."
                else:
                    buff += _str_list_compact(self[i].tolist()) + ""
                if ni < len(indices) - 1:
                    buff += "\n"
            return buff + ")"

        return f"nt.Tensor(shape={self.shape}, {self.numel} x <{self.dtype.name}>)"

    def __len__(self):
        if not len(self._shape):
            return 0
        return self._shape[0]

    def __getitem__(self, index: TensorIndex | tuple[TensorIndex, ...]) -> Tensor:
        if not isinstance(index, tuple):
            index = (index,)
        if self.ndim == 0:
            raise IndexError("Cannot index a scalar value.")
        if len([x for x in index if x is not None]) > self.ndim:
            raise IndexError(f"Too many indices ({len(index)}) for {self.ndim}D array.")

        index = expand_ellipsis(index, self.shape)
        view, new_index = _newview_indexing(self, index)

        if is_contiguous_view(index):
            # New view, same storage
            return Tensor._new_view(
                dtype=self.dtype,
                shape=view.shape,
                storage=self._data,
                strides=view._strides,
                offset=view._offset,
            )

        # Sequence/masked axes necessitate mem copy
        fax = _FancyAxes.compute(view.shape, new_index)
        new_data = _C.gather_from_axes(
            x=view._C_view,
            new_sh=fax.new_shape,
            fancy_dims_in_src=fax.fancy_dims_in_src,
            fancy_dims_data=[t._C_view for t in fax.fancy_dims_data],
            out_axis_is_fancy=fax.out_axis_is_fancy,
            out_axis_target=fax.out_axis_target,
        )
        return Tensor._new_contiguous(
            dtype=self.dtype, shape=fax.new_shape, storage=new_data
        )

    def __setitem__(
        self, index: TensorIndex | tuple[TensorIndex, ...], value: Tensor | InputType
    ) -> None:
        if not isinstance(index, tuple):
            index = (index,)
        if self.ndim == 0:
            raise IndexError("Cannot index into a scalar tensor.")
        if len([x for x in index if x is not None]) > self.ndim:
            raise IndexError(f"Too many indices ({len(index)}) for {self.ndim}D array.")

        index = expand_ellipsis(index, self.shape)
        view, new_index = _newview_indexing(self, index)

        if not isinstance(value, Tensor):
            value = Tensor(value, dtype=self.dtype)
        elif value.dtype != self.dtype:
            value = value.to(self.dtype)

        if is_contiguous_view(index):
            try:
                value = value.expand(view.shape)
            except ValueError as e:
                raise IndexError(str(e)) from e
            if value._data is self._data:  # aliasing guard
                value = value._to_contiguous()
            _C.copy_view(
                src=value._C_view,
                dst=_C.TensorView(self._data, view.shape, view._strides, view._offset),
            )
            return

        fax = _FancyAxes.compute(view.shape, new_index)
        try:
            value = value.expand(fax.new_shape)
        except ValueError as e:
            raise IndexError(str(e)) from e
        if value._data is self._data:  # aliasing guard
            value = value._to_contiguous()

        _C.scatter_to_axes(
            src=value._C_view,
            dst=self._C_view,
            fancy_dims_in_src=fax.fancy_dims_in_src,
            fancy_dims_data=[t._C_view for t in fax.fancy_dims_data],
            out_axis_is_fancy=fax.out_axis_is_fancy,
            out_axis_target=fax.out_axis_target,
        )

    @property
    def dtype(self) -> DataType:
        """Tensor internal data type."""
        return self._dtype

    @property
    def shape(self) -> TensorShape:
        """Tensor shape as an integer tuple (row-first)."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Returns the number of tensor dimensions."""
        return len(self._shape)

    @property
    def numel(self) -> int:
        """Returns the number of values in the tensor."""
        return math.prod(self.shape)

    @property
    def is_empty(self) -> bool:
        """Detects empty tensor."""
        return self.numel == 0

    @property
    def T(self) -> Tensor:
        """Returns a transposed view (no copy)."""
        if len(self.shape) < 2:
            return self
        return Tensor._new_view(
            self.dtype,
            tuple(reversed(self.shape)),
            self._data,
            tuple(reversed(self._strides)),
            self._offset,
        )

    def to(self, target: DataType) -> Tensor:
        """Cast the tensor to a new DataType, leaves inplace if no cast needed."""
        if target == self.dtype:
            return self
        new_buffer = _C.cast(self._data, target.cpp_dtype)
        return Tensor._new_view(
            target, self.shape, new_buffer, self._strides, self._offset
        )

    def reshape(self, *dims: int) -> Tensor:
        """Returns a reshaped view (no copy)."""
        if math.prod(dims) != self.numel:
            raise ValueError(f"Can't reshape {self.shape} into {dims}.")
        if len(dims) == 0:
            raise ValueError("No dimensions provided.")
        if dims == self.shape:
            return self
        base = self._to_contiguous()
        return Tensor._new_view(
            base.dtype, tuple(dims), base._data, strides=None, offset=base._offset
        )

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """Permutes two tensor dimensions."""
        if len(self.shape) < 2:
            return self
        shape, strides = list(self.shape), list(self._strides)
        shape[dim0], shape[dim1], strides[dim0], strides[dim1] = (
            shape[dim1],
            shape[dim0],
            strides[dim1],
            strides[dim0],
        )
        return Tensor._new_view(
            self.dtype,
            tuple(shape),
            self._data,
            tuple(strides),
            self._offset,
        )

    def expand(self, shape: TensorShape) -> Tensor:
        """Expand the tensor to target shape, no copy."""
        if len(shape) < self.ndim:
            raise ValueError(f"Cannot broadcast {self.shape} to {shape}.")
        strides = list(self._strides)
        for i in range(len(shape)):
            if i >= self.ndim:
                strides = [0] + strides
            else:
                src, tgt = self.shape[-i - 1], shape[-i - 1]
                if src == tgt:
                    continue
                elif src == 1:
                    strides[-i - 1] = 0  # broadcast
                else:
                    raise ValueError(
                        f"Cannot broadcast {self.shape} to {shape} (mismatch)."
                    )
        return Tensor._new_view(
            self.dtype, shape, self._data, tuple(strides), self._offset
        )

    def flatten(self) -> Tensor:
        """Flattens into a 1D tensor."""
        return self.reshape(self.numel)

    def equals(self, other: Tensor) -> bool:
        """Test per-coefficient equality, auto-promotes to best dtype."""
        if self.shape != other.shape:
            return False
        if self.is_empty and other.is_empty:
            return True
        promote_type = promote_dtypes(self.dtype, other.dtype)
        first = self
        if promote_type != self.dtype:
            first = self.to(promote_type)
        if promote_type != other.dtype:
            other = other.to(promote_type)
        return _C.equals(first._C_view, other._C_view)

    def tolist(self) -> InputType:
        """Converts the tensor to a list, preserving shape and type."""
        storage = memoryview(self._data)

        def rec(
            shape: TensorShape, strides: TensorShape, offset: int, depth: int
        ) -> InputType:
            if depth == len(shape):
                if self.dtype.is_bool:
                    return bool(storage[offset])
                return storage[offset]
            return [
                rec(shape, strides, offset + strides[depth] * i, depth + 1)
                for i in range(shape[depth])
            ]

        return rec(self.shape, self._strides, self._offset, 0)

    def item(self) -> bool | int | float:
        """Converts a single element tensor to a scalar."""
        if self.numel != 1:
            raise RuntimeError(f"Cannot convert {self} to a scalar.")
        storage = memoryview(self._data)
        return storage[self._offset]

    # Tensor ops

    def __add__(self, other: Tensor | float | int | bool) -> Tensor:
        return _binary_kernel_op(self, other, _C.add)

    def __radd__(self, other: Tensor | float | int | bool) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Tensor | float | int | bool) -> Tensor:
        return _binary_kernel_op(self, other, _C.subtract)

    def __rsub__(self, other: Tensor | float | int | bool) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other: Tensor | float | int | bool) -> Tensor:
        return _binary_kernel_op(self, other, _C.multiply)

    def __rmul__(self, other: Tensor | float | int | bool) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Tensor | float | int | bool) -> Tensor:
        return _binary_kernel_op(self, other, _C.divide)

    def __rtruediv__(self, other: Tensor | float | int | bool) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__truediv__(self)

    def __neg__(self) -> Tensor:
        return self.__mul__(-1)

    def __matmul__(self, other: Tensor) -> Tensor:
        this = self
        if this.ndim != 2 or other.ndim != 2:
            raise ValueError("Only 2D matmul is supported.")
        if this.shape[1] != other.shape[0]:
            raise ValueError(f"Cannot multiply {this.shape} @ {other.shape}.")
        shape = (this.shape[0], other.shape[1])
        dtype = promote_dtypes(this.dtype, other.dtype)
        this = this.to(dtype)
        other = other.to(dtype)
        return Tensor._new_contiguous(
            dtype=dtype, shape=shape, storage=_C.matmul(this._C_view, other._C_view)
        )

    def __rmatmul__(self, other: Tensor) -> Tensor:
        return other.__matmul__(self)

    def sum(
        self,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        *,
        dtype: DataType | None = None,
    ) -> Tensor:
        """Compute per-coef sum of tensor coefficients."""
        if axis is None:
            axis = tuple(range(self.ndim))
        elif isinstance(axis, int):
            axis = (axis,)
        axis = tuple(ax + self.ndim if ax < 0 else ax for ax in axis)
        if any(not isinstance(ax, int) for ax in axis):
            raise TypeError(f"Invalid axis: {axis}")
        if len(set(axis)) != len(axis):
            raise ValueError(f"Redundant axes: {axis}")
        if any(ax < 0 or ax >= self.ndim for ax in axis):
            raise IndexError(f"Invalid axis: {axis}")
        if keepdim:
            new_shape = tuple(1 if i in axis else s for i, s in enumerate(self.shape))
        else:
            new_shape = tuple(s for i, s in enumerate(self.shape) if i not in axis)

        if dtype is None:
            dtype = max(self.dtype, DataType.INT64)

        if self.is_empty:
            return Tensor._new_contiguous(
                dtype, new_shape, _C.zeros(new_shape, dtype.cpp_dtype)
            )

        return Tensor._new_contiguous(
            dtype=dtype,
            shape=new_shape,
            storage=_C.sum(self._C_view, axis, keepdim, dtype.cpp_dtype),
        )

    def mean(
        self,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        *,
        dtype: DataType | None = None,
    ) -> Tensor:
        """Compute per-coef sum of tensor coefficients."""
        if dtype is None:
            dtype = self.dtype
        if dtype < DataType.FP32:
            raise TypeError("Cannot take mean of integer tensor, cast to fp.")
        sum_ = self.sum(axis=axis, keepdim=keepdim, dtype=dtype)
        return sum_ / (self.numel // sum_.numel)

    def exp(self) -> Tensor:
        """Apply coefficient-wise exponential."""
        if self.is_empty:
            return self
        dtype = promote_dtypes(self.dtype, DataType.FP32)
        return Tensor._new_contiguous(dtype, self.shape, _C.exp(self.to(dtype)._C_view))

    def log(self) -> Tensor:
        """Apply coefficient-wise exponential."""
        if self.is_empty:
            return self
        dtype = promote_dtypes(self.dtype, DataType.FP32)
        return Tensor._new_contiguous(dtype, self.shape, _C.log(self.to(dtype)._C_view))

    def pow(self, exponent: float | int | bool | Tensor) -> Tensor:
        """Apply coefficient-wise exponential."""
        if self.is_empty:
            return self
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)
        dtype = promote_dtypes(self.dtype, exponent.dtype)
        exponent_fp = float(exponent.item())
        if exponent_fp < 0 and dtype <= DataType.INT64:
            raise RuntimeError("Cannot use negative exponent with integer tensor.")
        return Tensor._new_contiguous(
            dtype, self.shape, _C.pow(self.to(dtype)._C_view, exponent_fp)
        )

    def __pow__(self, other: Tensor | float | int | bool) -> Tensor:
        return self.pow(other)

    def __rpow__(self, other: Tensor | float | int | bool) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.pow(self)

    # Private operators

    def _is_contiguous(self) -> bool:
        """Checks if current view is contiguous in memory."""
        return self._strides == _infer_strides(self._shape)

    def _to_contiguous(self) -> Tensor:
        """Returns a contiguous version of the tensor (copy if necessary)."""
        if self._is_contiguous():
            return self
        return Tensor(self.tolist(), self.dtype)

    @property
    def _C_view(self) -> _C.TensorView:
        """Return a C++-compatible tensor view."""
        return _C.TensorView(self._data, self.shape, self._strides, self._offset)


# Private functions


def _extract_tensor_data(
    data: InputType, user_dtype: DataType | None = None
) -> tuple[DataType, TensorShape, _C.Storage]:
    """Automatically extract tensor information."""

    # Accumulators
    shape: list[int] = []
    flat: list[float | int | bool] = []
    auto_dtype: DataType | None = None
    leaf_level: int | None = None

    def rec(data: InputType, level: int) -> None:
        # Recursive walk through the data
        nonlocal auto_dtype
        nonlocal leaf_level

        if level > MAX_TENSOR_LEVEL:
            raise ValueError(f"Maximum tensor depth reached: {MAX_TENSOR_LEVEL}")
        if isinstance(data, (bool, int, float)):
            if leaf_level is None:
                leaf_level = level
            elif level != leaf_level:
                raise ValueError("Leaf found at non-terminal level.")
            dtype = DataType.from_type(data)
            if auto_dtype is None or dtype > auto_dtype:
                auto_dtype = dtype
            flat.append(data)
        elif isinstance(data, list):
            nnodes = len(data)
            if level >= len(shape):
                shape.append(nnodes)
            elif shape[level] != nnodes:
                raise ValueError(
                    f"Unhomogeneous tensor shape detected ({shape[level]} != {nnodes})"
                )
            for node in data:
                rec(node, level + 1)
        else:
            raise TypeError(f"Unsupported tensor element type: {type(data)}")

    rec(data, 0)

    # Final type casting
    auto_dtype = auto_dtype or DataType.FP32
    if user_dtype is not None:
        final_dtype = user_dtype
    else:
        final_dtype = auto_dtype

    # Needs int cast
    if final_dtype.is_bool:
        values = list(bool(x) for x in flat)
    elif auto_dtype.is_float and final_dtype.is_int:
        values = list(int(x) for x in flat)
    else:
        values = flat

    return (
        final_dtype,
        tuple(shape),
        _C.Storage.from_iterable(values, final_dtype.cpp_dtype),
    )


def _infer_strides(shape: TensorShape) -> TensorShape:
    """Infer major strides from shape."""
    strides = []
    for i in range(len(shape)):
        if i == 0:
            new_stride = 1
        else:
            new_stride = strides[0] * shape[-i]
        strides.insert(0, new_stride)
    return tuple(strides)


def _str_list_compact(li: Any) -> str:
    """Returns a compact list string."""
    if not isinstance(li, list):
        return str(li)
    if len(li) < 8:
        return str(li)
    return (
        "["
        + ", ".join(str(x) for x in li[:3])
        + ", ..., "
        + ", ".join(str(x) for x in li[-3:])
        + "]"
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


@dataclass
class _FancyAxes:
    """Container for get_fancy_axes helper."""

    new_shape: TensorShape
    fancy_dims_in_src: list[int]
    fancy_dims_data: list[Tensor]
    out_axis_is_fancy: list[bool]
    out_axis_target: list[int]

    @classmethod
    def compute(cls, shape: TensorShape, index: tuple[TensorIndex, ...]) -> _FancyAxes:
        fancy_dims_in_src: list[int] = []
        basic_axes: list[int] = []
        basic_shape: list[int] = []
        indarrs_raw: list[Tensor] = []

        for dim in range(len(shape)):
            # Non-fancy axis, skip
            if (
                dim >= len(index)
                or index[dim] is None
                or isinstance(index[dim], (int, slice))
            ):
                basic_axes.append(dim)
                basic_shape.append(shape[dim])
                continue

            # Fancy axis (integer-based or boolean mask)
            fancy_index = index[dim]
            if isinstance(fancy_index, Tensor):
                pass
            elif isinstance(fancy_index, Sequence):
                try:
                    fancy_index = Tensor(fancy_index)  # type: ignore
                except Exception as e:
                    raise IndexError(
                        f"Could not interpret index {dim} as a Tensor: {e}"
                    )
            else:
                raise IndexError(f"Unexpected selector type {type(fancy_index)}")

            # TODO: do it index-based once implemented
            content = memoryview(fancy_index._data)
            for i in range(len(content)):
                if content[i] < 0:
                    content[i] = content[i] + shape[dim]
                if content[i] < 0 or content[i] >= shape[dim]:
                    raise IndexError(
                        f"Invalid index {content[i]} in axis of len {shape[dim]}."
                    )

            if fancy_index.dtype == DataType.BOOL:
                if fancy_index.ndim > 1:
                    raise NotImplementedError("nD boolean masks not here (yet).")
                if len(fancy_index) != shape[dim]:
                    raise IndexError(
                        f"Wrong boolean index shape: {len(fancy_index)} (expected {shape[dim]})."
                    )
                indices = [i for i, b in enumerate(content) if b]
                fancy_index = Tensor(indices)

            fancy_dims_in_src.append(dim)
            indarrs_raw.append(fancy_index)

        indarr_shape = broadcast_shapes(*(t.shape for t in indarrs_raw))
        fancy_dims_data = [t.expand(indarr_shape) for t in indarrs_raw]

        # compute new shape and axes plan
        nfancy = len(fancy_dims_in_src)
        nidx = len(indarr_shape)
        axes_diff = [
            fancy_dims_in_src[i + 1] - fancy_dims_in_src[i] for i in range(nfancy - 1)
        ]

        out_axis_is_fancy = []
        out_axis_target = []
        if nfancy > 0 and any(d != 1 for d in axes_diff):  # Not contiguous case
            new_shape = tuple(list(indarr_shape) + basic_shape)
            for i in range(len(new_shape)):
                if i < nidx:
                    out_axis_is_fancy.append(True)
                    out_axis_target.append(i)
                else:
                    out_axis_is_fancy.append(False)
                    out_axis_target.append(basic_axes[i - nidx])
        else:
            a0 = fancy_dims_in_src[0]
            new_shape = tuple(basic_shape[:a0] + list(indarr_shape) + basic_shape[a0:])
            for i in range(len(new_shape)):
                if i < a0:
                    out_axis_is_fancy.append(False)
                    out_axis_target.append(basic_axes[i])
                elif i < a0 + nidx:
                    out_axis_is_fancy.append(True)
                    out_axis_target.append(i - a0)
                else:
                    out_axis_is_fancy.append(False)
                    out_axis_target.append(basic_axes[i - nidx])

        return _FancyAxes(
            new_shape=new_shape,
            fancy_dims_in_src=fancy_dims_in_src,
            fancy_dims_data=fancy_dims_data,
            out_axis_is_fancy=out_axis_is_fancy,
            out_axis_target=out_axis_target,
        )


def _newview_indexing(
    x: Tensor, index: tuple[TensorIndex, ...]
) -> tuple[Tensor, tuple[TensorIndex, ...]]:
    """Indexes a new view of the same storage (simple indexing)."""
    offset = x._offset
    new_shape: list[int] = []
    new_strides: list[int] = []
    new_index: list[TensorIndex] = []

    olddim = 0
    newdim = 0
    while olddim < len(x.shape) or newdim < len(index):
        if newdim >= len(index) or (
            (not isinstance(index[newdim], (int, slice)))
            and (index[newdim] is not None)
        ):
            new_shape.append(x.shape[olddim])
            new_strides.append(x._strides[olddim])
            new_index.append(None if newdim >= len(index) else index[newdim])
            olddim += 1
            newdim += 1
            continue

        sel = index[newdim]
        if sel is None:
            new_shape.append(1)
            new_strides.append(0)
            new_index.append(None)
            newdim += 1
        else:
            nelem = x.shape[olddim]
            if isinstance(sel, int):
                if sel < 0:
                    sel += nelem
                if sel < 0 or sel >= nelem:
                    raise IndexError(
                        f"Cannot index at position {sel} in {nelem} elements."
                    )
                offset += sel * x._strides[olddim]
            elif isinstance(sel, slice):
                start, stop, step = sel.indices(nelem)
                length = len(range(start, stop, step))
                new_shape.append(length)
                new_strides.append(x._strides[olddim] * step)
                new_index.append(None)
                offset += start * x._strides[olddim]
            else:
                raise ValueError(f"Unhandle index type: {type(sel)}.")
            olddim += 1
            newdim += 1

    return Tensor._new_view(
        x.dtype, tuple(new_shape), x._data, tuple(new_strides), offset
    ), tuple(new_index)


TensorIndex = (
    int | slice | Sequence[bool | int | Sequence[int]] | EllipsisType | Tensor | None
)
TensorLike = Tensor | float | int | bool
