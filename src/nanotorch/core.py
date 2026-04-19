"""Base tensor class."""

import math
from enum import Enum

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
        max_index = offset + sum(
            (s - 1) * abs(st) for s, st in zip(shape, strides) if s > 0
        )
        max_storage = len(memoryview(flat_data))
        if math.prod(shape) > 0 and max_index >= max_storage:
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
        return f"nt.Tensor({self.tolist()})"

    def __len__(self):
        if not len(self._shape):
            return 0
        return self._shape[0]

    def __getitem__(self, index: int) -> "Tensor":
        if index < 0:
            index = len(self) + index
        if len(self.shape) == 0 or index < 0 or index >= len(self):
            raise IndexError(
                f"Cannot access index {index} in tensor of shape {self.shape}."
            )
        shape = self.shape[1:]
        offset = self._offset + self._strides[0] * index
        return Tensor.init_from_components(
            dtype=self.dtype,
            shape=shape,
            flat_data=self._data,
            strides=self._strides[1:],
            offset=offset,
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
        if math.prod(dims) != math.prod(self.shape):
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

    # Common ops shortcuts

    def sum(self) -> "Tensor":
        """Compute per-coef sum of tensor coefficients."""
        if self.is_empty:  # FIXME: strides&offset
            return Tensor(0, dtype=self.dtype)
        return Tensor(_C.sum(self._data, self.shape, self._strides, self._offset))

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
        return _C.equals(
            first._data,
            first.shape,
            first._strides,
            first._offset,
            other._data,
            other.shape,
            other._strides,
            other._offset,
        )

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
