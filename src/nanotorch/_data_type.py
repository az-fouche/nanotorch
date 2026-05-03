"""Handles tensor values typing."""

from nanotorch import _C

InputType = list | float | int | bool

Dtype = _C.Dtype


def dtype_from_type(data: InputType) -> Dtype:
    """Automatically initializes a Dtype from item."""
    if isinstance(data, bool):
        return Dtype.Bool
    if isinstance(data, int):
        return Dtype.Int64
    if isinstance(data, float):
        return Dtype.Float32
    raise TypeError(f"Unsupported Tensor value type: {type(data)}")


def is_bool(x: Dtype) -> bool:
    return x == Dtype.Bool


def is_int(self) -> bool:
    return self in (Dtype.Int32, Dtype.Int64)


def is_float(self) -> bool:
    return self in (Dtype.Float32, Dtype.Float64)


def promote_dtypes(*dtypes: Dtype) -> Dtype:
    """Compute the highest precision Dtype."""
    if not dtypes:
        return Dtype.Float32
    return max(dtypes)  # type: ignore


# Syntactic sugar (nt.bool_)
bool_ = Dtype.Bool
int32 = Dtype.Int32
int64 = Dtype.Int64
float32 = Dtype.Float32
float64 = Dtype.Float64
