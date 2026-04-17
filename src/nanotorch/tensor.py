"""Base tensor class."""

from array import array
from enum import Enum, auto

InputType = list | float | int | bool
TensorShape = tuple[int, ...]

MAX_TENSOR_DEPTH = 2


class DataType(Enum):
    """Supported tensor data types."""

    BOOL = auto()
    INT32 = auto()
    INT64 = auto()
    FP32 = auto()
    FP64 = auto()

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
        return self._dtype

    @property
    def shape(self) -> TensorShape:
        return self._shape


def _extract_tensor_data(
    data: InputType, user_dtype: DataType | None = None
) -> tuple[DataType, TensorShape, array]:
    """Automatically extract tensor information."""

    shape: list[int] = []
    flat: list[float | int | bool] = []
    auto_dtype: DataType | None = None

    def rec(data: InputType, depth: int) -> None:
        nonlocal auto_dtype

        if depth > MAX_TENSOR_DEPTH:
            raise ValueError(f"Maximum tensor depth reached: {MAX_TENSOR_DEPTH}")
        if isinstance(data, (bool, int, float)):
            dtype = DataType.from_type(data)
            if (
                auto_dtype is None
                or dtype == DataType.FP32
                or dtype == DataType.INT64
                and auto_dtype != DataType.FP32
            ):
                auto_dtype = dtype
            flat.append(data)
        elif isinstance(data, list):
            shape.append(len(data))
            for node in data:
                rec(node, depth + 1)

    rec(data, 0)

    if len(shape) > 1:
        nrows = shape[0]
        if nrows + 1 != len(shape):
            raise ValueError(
                f"Incompatible shape: expected {nrows} rows, got {len(shape) - 1}."
            )
        if any(s != shape[1] for s in shape[2:]):
            raise ValueError("Incompatible shape: unhomogeneous rows length.")
        shape = shape[:2]

    if auto_dtype is None:
        auto_dtype = DataType.FP32
    if user_dtype is not None:
        final_dtype = user_dtype
    else:
        final_dtype = auto_dtype

    match final_dtype:
        case DataType.BOOL:
            dtype_code = "b"
        case DataType.INT32:
            dtype_code = "l"
        case DataType.INT64:
            dtype_code = "q"
        case DataType.FP32 | None:
            dtype_code = "f"
        case DataType.FP64:
            dtype_code = "d"
        case _:
            raise TypeError(f"Unhandled data type: {DataType}")

    # Needs int cast
    if auto_dtype in (DataType.FP32, DataType.FP64) and final_dtype in (
        DataType.BOOL,
        DataType.INT32,
        DataType.INT64,
    ):
        values = (int(x) for x in flat)
    else:
        values = flat

    return final_dtype, tuple(shape), array(dtype_code, values)


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
