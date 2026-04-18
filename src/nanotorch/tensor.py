"""Base tensor class."""

from enum import Enum

from nanotorch import _C

InputType = list | float | int | bool
TensorShape = tuple[int, ...]

MAX_TENSOR_DEPTH = 2


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

    @property
    def dtype(self) -> DataType:
        """Tensor internal data type."""
        return self._dtype

    @property
    def shape(self) -> TensorShape:
        """Tensor shape as an integer tuple (row-first)."""
        return self._shape

    @property
    def is_empty(self) -> bool:
        """Detects empty tensor."""
        return self.shape == (0,)

    def to(self, target: DataType) -> "Tensor":
        """Cast the tensor to a new DataType, leaves inplace if no cast needed."""
        if target == self.dtype:
            return self
        new_buffer = _C.cast(self._data, target.cpp_dtype)
        return Tensor._from_tensor_components(target, self.shape, new_buffer)

    def sum(self) -> "Tensor":
        """Compute per-coef sum of tensor coefficients."""
        if self.is_empty:
            return Tensor(0, dtype=self.dtype)
        return Tensor(_C.sum(self._data))

    def equals(self, other: "Tensor") -> bool:
        """Test per-coefficient equality, auto-promotes to best dtype."""
        if self.shape != other.shape:  # FIXME: strides&offset
            return False
        if self.is_empty and other.is_empty:
            return True
        promote_type = _promote_types(self.dtype, other.dtype)
        if promote_type != other.dtype:
            other = other.to(promote_type)
        if promote_type != self.dtype:
            return _C.equals(self.to(promote_type)._data, other._data)
        return _C.equals(self._data, other._data)

    def tolist(self) -> InputType:
        """Converts the tensor to a list, preserving shape and type."""
        storage = memoryview(self._data).tolist()
        if self.dtype.is_bool:
            storage = [bool(x) for x in storage]
        if len(self.shape) == 0:
            return storage[self._offset]
        if len(self.shape) == 1:
            return [
                storage[self._offset + self._strides[0] * i]
                for i in range(self.shape[0])
            ]
        return [
            [
                storage[self._offset + self._strides[0] * i + self._strides[1] * j]
                for j in range(self.shape[1])
            ]
            for i in range(self.shape[0])
        ]

    @classmethod
    def _from_tensor_components(
        cls, dtype: DataType, shape: TensorShape, flat_data: _C.Storage
    ) -> "Tensor":
        """Intializes a new tensor from precomputed components."""
        new = cls.__new__(cls)
        new._dtype = dtype
        new._shape = shape
        new._data = flat_data
        new._strides = _infer_strides(shape)
        new._offset = 0
        return new


# Function-style ops


def tensor(data: InputType, dtype: DataType | None = None) -> Tensor:
    """Initialize a new tensor."""
    return Tensor(data, dtype)


def sum(tensor: Tensor) -> Tensor:
    """Compute per-coef sum of tensor coefficients."""
    return tensor.sum()


def equals(t1: Tensor, t2: Tensor) -> bool:
    """Test per-coefficient equality, auto-promotes to best dtype."""
    return t1.equals(t2)


# Private functions


def _extract_tensor_data(
    data: InputType, user_dtype: DataType | None = None
) -> tuple[DataType, TensorShape, _C.Storage]:
    """Automatically extract tensor information."""

    # Accumulators
    shape: list[int] = []
    flat: list[float | int | bool] = []
    auto_dtype: DataType | None = None

    def rec(data: InputType, depth: int) -> None:
        # Recursive walk through the data
        nonlocal auto_dtype

        if depth > MAX_TENSOR_DEPTH:
            raise ValueError(f"Maximum tensor depth reached: {MAX_TENSOR_DEPTH}")
        if isinstance(data, (bool, int, float)):
            dtype = DataType.from_type(data)
            if auto_dtype is None or dtype > auto_dtype:
                auto_dtype = dtype
            flat.append(data)
        elif isinstance(data, list):
            shape.append(len(data))
            for node in data:
                rec(node, depth + 1)
        else:
            raise TypeError(f"Unsupported tensor element type: {type(data)}")

    rec(data, 0)

    # Shape validation
    if len(shape) > 1:
        nrows = shape[0]
        if nrows + 1 != len(shape):
            raise ValueError(
                f"Incompatible shape: expected {nrows} rows, got {len(shape) - 1}."
            )
        if any(s != shape[1] for s in shape[2:]):
            raise ValueError("Incompatible shape: unhomogeneous rows length.")
        shape = shape[:2]

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
