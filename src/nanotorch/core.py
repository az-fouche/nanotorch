"""Base tensor class."""

from __future__ import annotations

import math
from enum import Enum
from types import EllipsisType
from typing import Any, NamedTuple, Sequence

from nanotorch import _C

InputType = list | float | int | bool
TensorShape = tuple[int, ...]

MAX_TENSOR_LEVEL = 32


class DataType(Enum):
    """Supported tensor data types."""

    # Promotion priority
    BOOL = 0
    INT32 = 1
    INT64 = 2
    FP32 = 3
    FP64 = 4

    @classmethod
    def from_type(cls, data: InputType) -> "DataType":
        """Automatically initializes a DataType from item."""
        if isinstance(data, bool):
            return DataType.BOOL
        if isinstance(data, int):
            return DataType.INT64
        if isinstance(data, float):
            return DataType.FP32
        raise TypeError(f"Unsupported Tensor value type: {type(data)}")

    @property
    def cpp_dtype(self) -> _C.Dtype:
        match self:
            case DataType.BOOL:
                return _C.Dtype.Bool
            case DataType.INT32:
                return _C.Dtype.Int32
            case DataType.INT64:
                return _C.Dtype.Int64
            case DataType.FP32:
                return _C.Dtype.Float32
            case DataType.FP64:
                return _C.Dtype.Float64
            case _:
                raise TypeError(f"Unhandled data type by C++: {DataType}")

    @classmethod
    def from_cpp(cls, cpptype: _C.Dtype):
        match cpptype:
            case _C.Dtype.Bool:
                return DataType.BOOL
            case _C.Dtype.Int32:
                return DataType.INT32
            case _C.Dtype.Int64:
                return DataType.INT64
            case _C.Dtype.Float32:
                return DataType.FP32
            case _C.Dtype.Float64:
                return DataType.FP64
            case _:
                raise TypeError(f"Unhandled data type by C++: {DataType}")

    # Promotion sugar
    def __gt__(self, other: "DataType") -> bool:
        return self.value > other.value

    def __lt__(self, other: "DataType") -> bool:
        return self.value < other.value

    def __ge__(self, other: "DataType") -> bool:
        return self.value >= other.value

    def __le__(self, other: "DataType") -> bool:
        return self.value <= other.value

    @property
    def is_bool(self) -> bool:
        return self == DataType.BOOL

    @property
    def is_int(self) -> bool:
        return self in (DataType.INT32, DataType.INT64)

    @property
    def is_float(self) -> bool:
        return self in (DataType.FP32, DataType.FP64)


bool_ = DataType.BOOL
int32 = DataType.INT32
int64 = DataType.INT64
float32 = DataType.FP32
float64 = DataType.FP64


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

        dtype, shape, flat_data = _extract_tensor_data(data, dtype)

        self._dtype = dtype
        self._shape = shape
        self._data = flat_data
        self._strides = _infer_strides(shape)
        self._offset = 0

    @classmethod
    def init_from_components(
        cls,
        dtype: DataType,
        shape: TensorShape,
        flat_data: _C.Storage,
        strides: TensorShape | None,
        offset: int | None,
    ) -> "Tensor":
        """Intializes a new tensor from precomputed components."""

        strides = _infer_strides(shape) if strides is None else strides
        offset = 0 if offset is None else offset
        min_index = offset + sum(
            (s - 1) * st for s, st in zip(shape, strides) if st < 0
        )
        max_index = offset + sum(
            (s - 1) * st for s, st in zip(shape, strides) if st > 0
        )
        max_storage = len(memoryview(flat_data))
        if math.prod(shape) > 0 and (min_index < 0 or max_index >= max_storage):
            raise IndexError(
                f"View max index {max_index} exceeds storage capacity {max_storage}."
            )

        new = cls.__new__(cls)
        new._dtype = dtype
        new._shape = shape
        new._data = flat_data
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

        return f"nt.Tensor(shape={self.shape}, {self.numel} x <{self.dtype.name}>"

    def __len__(self):
        if not len(self._shape):
            return 0
        return self._shape[0]

    def __getitem__(self, index: TensorIndex | tuple[TensorIndex, ...]) -> "Tensor":
        if not isinstance(index, tuple):
            index = (index,)
        if self.ndim == 0:
            raise IndexError("Cannot index a scalar value.")
        if len([x for x in index if x is not None]) > self.ndim:
            raise IndexError(f"Too many indices ({len(index)}) for {self.ndim}D array.")
        index = _expand_ellipsis(index, self.shape)

        shape, strides, offset, new_index = _newview_indexing(
            index, self.shape, self._strides, self._offset
        )

        if _is_contiguous_view(index):
            # New view, same storage
            return Tensor.init_from_components(
                dtype=self.dtype,
                shape=shape,
                flat_data=self._data,
                strides=strides,
                offset=offset,
            )

        # Sequence/masked axes necessitate mem copy
        fax = _get_fancy_axes(shape, new_index)
        new_data = _C.gather_from_axes(
            x=Tensor.init_from_components(
                self.dtype,
                shape,
                self._data,
                strides,
                offset,
            )._C_tensor_view(),
            new_sh=fax.new_shape,
            fancy_dims_in_src=fax.fancy_dims_in_src,
            fancy_dims_data=[t._C_tensor_view() for t in fax.fancy_dims_data],
            out_axis_is_fancy=fax.out_axis_is_fancy,
            out_axis_target=fax.out_axis_target,
        )
        return Tensor.init_from_components(
            dtype=self.dtype,
            shape=fax.new_shape,
            flat_data=new_data,
            strides=None,
            offset=None,
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
        return self.shape == (0,)

    @property
    def T(self) -> "Tensor":
        """Returns a transposed view (no copy)."""
        if len(self.shape) < 2:
            return self
        return Tensor.init_from_components(
            self.dtype,
            tuple(reversed(self.shape)),
            self._data,
            tuple(reversed(self._strides)),
            self._offset,
        )

    def to(self, target: DataType) -> "Tensor":
        """Cast the tensor to a new DataType, leaves inplace if no cast needed."""
        if target == self.dtype:
            return self
        new_buffer = _C.cast(self._data, target.cpp_dtype)
        return Tensor.init_from_components(
            target, self.shape, new_buffer, self._strides, self._offset
        )

    def reshape(self, *dims: int) -> "Tensor":
        """Returns a reshaped view (no copy)."""
        if math.prod(dims) != self.numel:
            raise ValueError(f"Can't reshape {self.shape} into {dims}.")
        if len(dims) == 0:
            raise ValueError("No dimensions provided.")
        if dims == self.shape:
            return self
        base = self._to_contiguous()
        return Tensor.init_from_components(
            base.dtype, tuple(dims), base._data, strides=None, offset=base._offset
        )

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
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
        return Tensor.init_from_components(
            self.dtype,
            tuple(shape),
            self._data,
            tuple(strides),
            self._offset,
        )

    def expand(self, shape: TensorShape) -> "Tensor":
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
        return Tensor.init_from_components(
            self.dtype, shape, self._data, tuple(strides), self._offset
        )

    # Common ops shortcuts

    def sum(self) -> "Tensor":
        """Compute per-coef sum of tensor coefficients."""
        if self.is_empty:  # FIXME: strides&offset
            return Tensor(0, dtype=self.dtype)
        return Tensor(_C.sum(self._C_tensor_view()))

    def equals(self, other: "Tensor") -> bool:
        """Test per-coefficient equality, auto-promotes to best dtype."""
        if self.shape != other.shape:
            return False
        if self.is_empty and other.is_empty:
            return True
        promote_type = _promote_types(self.dtype, other.dtype)
        first = self
        if promote_type != self.dtype:
            first = self.to(promote_type)
        if promote_type != other.dtype:
            other = other.to(promote_type)
        return _C.equals(first._C_tensor_view(), other._C_tensor_view())

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

    # Private operators

    def _is_contiguous(self) -> bool:
        """Checks if current view is contiguous in memory."""
        return self._strides == _infer_strides(self._shape)

    def _to_contiguous(self) -> "Tensor":
        """Returns a contiguous version of the tensor (copy if necessary)."""
        if self._is_contiguous():
            return self
        return Tensor(self.tolist(), self.dtype)

    def _C_tensor_view(self) -> _C.TensorView:
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


def _promote_types(*dtypes: DataType) -> DataType:
    """Compute the best suited promotion type."""
    best_dtype: DataType | None = None
    for dtype in dtypes:
        if best_dtype is None or best_dtype < dtype:
            best_dtype = dtype
            if best_dtype == DataType.FP64:
                return best_dtype
    return best_dtype or DataType.FP32


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


# Indexing machinery


class _FancyAxes(NamedTuple):
    """Container for _get_fancy_axes helper."""

    new_shape: TensorShape
    fancy_dims_in_src: list[int]
    fancy_dims_data: list[Tensor]
    out_axis_is_fancy: list[bool]
    out_axis_target: list[int]


def _expand_ellipsis(
    index: tuple[TensorIndex, ...], shape: TensorShape
) -> tuple[TensorIndex, ...]:
    """Expands ellipsis into slice [..., 0] -> [:, :, :, 0]"""
    if len(index) == 0:
        return ()
    if sum(isinstance(i, EllipsisType) for i in index) > 1:
        raise IndexError("Multiple ellipses detected.")
    ndim = len(shape)
    nind = sum(i is not None for i in index)
    if isinstance(index[0], EllipsisType):
        return tuple([slice(None)] * (ndim - nind + 1) + list(index[1:]))
    if isinstance(index[-1], EllipsisType):
        return tuple(list(index[:-1]) + [slice(None)] * (ndim - nind + 1))
    pointer = 0
    index_exp: list[TensorIndex] = []
    while pointer < len(index):
        elem = index[pointer]
        if not isinstance(elem, EllipsisType):
            index_exp.append(elem)
        else:
            index_exp.extend([slice(None)] * (ndim - nind + 1))
        pointer += 1
    return tuple(index_exp)


def _is_contiguous_view(index: tuple[TensorIndex, ...]) -> bool:
    """Computes if the indexing can be done without copy."""
    for idx in index:
        if not isinstance(idx, (int, slice)) and idx is not None:
            return False
    return True


def _newview_indexing(
    index: tuple[TensorIndex, ...],
    shape: TensorShape,
    strides: TensorShape,
    offset: int,
) -> tuple[TensorShape, TensorShape, int, tuple[TensorIndex, ...]]:
    """Indexes a new view of the same storage (simple indexing)."""
    new_shape: list[int] = []
    new_strides: list[int] = []
    new_index: list[TensorIndex] = []

    olddim = 0
    newdim = 0
    while olddim < len(shape) or newdim < len(index):
        if newdim >= len(index) or (
            (not isinstance(index[newdim], (int, slice)))
            and (index[newdim] is not None)
        ):
            new_shape.append(shape[olddim])
            new_strides.append(strides[olddim])
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
            nelem = shape[olddim]
            if isinstance(sel, int):
                if sel < 0:
                    sel += nelem
                if sel < 0 or sel >= nelem:
                    raise IndexError(
                        f"Cannot index at position {sel} in {nelem} elements."
                    )
                offset += sel * strides[olddim]
            elif isinstance(sel, slice):
                start, stop, step = sel.indices(nelem)
                length = len(range(start, stop, step))
                new_shape.append(length)
                new_strides.append(strides[olddim] * step)
                new_index.append(None)
                offset += start * strides[olddim]
            else:
                raise ValueError(f"Unhandle index type: {type(sel)}.")
            olddim += 1
            newdim += 1

    return tuple(new_shape), tuple(new_strides), offset, tuple(new_index)


def _get_fancy_axes(shape: TensorShape, index: tuple[TensorIndex, ...]) -> _FancyAxes:
    """Compute fancy axes characteristics."""
    basic_axes: list[int] = []
    basic_shape: list[int] = []
    fancy_axes: list[int] = []
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
                raise IndexError(f"Could not interpret index {dim} as a Tensor: {e}")
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
            indices = [i for i, b in enumerate(content) if b]
            fancy_index = Tensor(indices)

        fancy_axes.append(dim)
        indarrs_raw.append(fancy_index)

    indarr_shape = _broadcast_shapes(*(t.shape for t in indarrs_raw))
    indarrs = [t.expand(indarr_shape) for t in indarrs_raw]

    # compute new shape and axes plan
    nfancy = len(fancy_axes)
    nidx = len(indarr_shape)
    axes_diff = [fancy_axes[i + 1] - fancy_axes[i] for i in range(nfancy - 1)]

    is_fancy = []
    fancy_target = []
    if nfancy > 0 and any(d != 1 for d in axes_diff):  # Not contiguous case
        new_shape = tuple(list(indarr_shape) + basic_shape)
        for i in range(len(new_shape)):
            if i < nidx:
                is_fancy.append(True)
                fancy_target.append(i)
            else:
                is_fancy.append(False)
                fancy_target.append(basic_axes[i - nidx])
    else:
        a0 = fancy_axes[0]
        new_shape = tuple(basic_shape[:a0] + list(indarr_shape) + basic_shape[a0:])
        for i in range(len(new_shape)):
            if i < a0:
                is_fancy.append(False)
                fancy_target.append(basic_axes[i])
            elif i < a0 + nidx:
                is_fancy.append(True)
                fancy_target.append(i - a0)
            else:
                is_fancy.append(False)
                fancy_target.append(basic_axes[i - nidx])

    return _FancyAxes(
        new_shape=new_shape,
        fancy_dims_data=indarrs,
        fancy_dims_in_src=fancy_axes,
        out_axis_is_fancy=is_fancy,
        out_axis_target=fancy_target,
    )


def _broadcast_shapes(*shapes: TensorShape) -> TensorShape:
    """Broadcast to the smallest compatible shape, raises if not possible."""
    if not shapes:
        return tuple()
    final_shape = list(shapes[0])
    for s in shapes[1:]:
        ldiff = len(final_shape) - len(s)
        if ldiff > 0:
            s = [1] * ldiff + list(s)
        elif ldiff < 0:
            final_shape = [1] * -ldiff + final_shape
        for i in range(len(final_shape)):
            if final_shape[i] == 1:
                final_shape[i] = s[i]
            elif s[i] == 1:
                continue
            elif final_shape[i] != s[i]:
                raise IndexError(f"Cannot broadcast shapes {s} and {final_shape}.")
    return tuple(final_shape)


TensorIndex = (
    int | slice | Sequence[bool | int | Sequence[int]] | EllipsisType | Tensor | None
)
