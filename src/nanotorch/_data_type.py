"""Handles tensor values typing."""

from enum import Enum

from nanotorch import _C

InputType = list | float | int | bool


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


def promote_dtypes(*dtypes: DataType) -> DataType:
    """Compute the highest precision DataType."""
    if not dtypes:
        return DataType.FP32
    return max(dtypes)  # type: ignore


bool_ = DataType.BOOL
int32 = DataType.INT32
int64 = DataType.INT64
float32 = DataType.FP32
float64 = DataType.FP64
