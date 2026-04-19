from nanotorch import _C

from .core import DataType, InputType, Tensor, TensorShape


def tensor(data: InputType, dtype: DataType | None = None) -> Tensor:
    """Initialize a new tensor."""
    return Tensor(data, dtype)


def zeros(shape: int | TensorShape, dtype: DataType = DataType.FP32) -> Tensor:
    """Initialize a new tensor filled with zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor.init_from_components(
        dtype, shape, _C.zeros(shape, dtype.cpp_dtype), None, None
    )


def ones(shape: int | TensorShape, dtype: DataType = DataType.FP32) -> Tensor:
    """Initialize a new tensor filled with ones."""
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor.init_from_components(
        dtype, shape, _C.ones(shape, dtype.cpp_dtype), None, None
    )


def full(
    shape: int | TensorShape, value: bool | int | float, dtype: DataType = DataType.FP32
) -> Tensor:
    """Initialize a new tensor filled with set value."""
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor.init_from_components(
        dtype, shape, _C.full(shape, value, dtype.cpp_dtype), None, None
    )


def eye(n: int, dtype: DataType = DataType.FP32) -> Tensor:
    """Initialize a new eye square tensor."""
    return Tensor.init_from_components(
        dtype, (n, n), _C.eye(n, dtype.cpp_dtype), None, None
    )


def arange(
    n: int, start: int = 0, step: int = 1, dtype: DataType = DataType.INT64
) -> Tensor:
    """Initialize a new tensor containing an arithmetic range."""
    return Tensor.init_from_components(
        dtype, (n,), _C.arange(n, start, step, dtype.cpp_dtype), None, None
    )
