import enum
from typing import Sequence

from typing_extensions import Buffer

class Dtype(enum.Enum):
    """C++-side storage types."""

    Bool = 0
    Int32 = 1
    Int64 = 2
    Float32 = 3
    Float64 = 4

# Layout contract: flattened array representations
class Storage(Buffer):
    @staticmethod
    def allocate(n: int, dtype: Dtype) -> "Storage": ...
    @staticmethod
    def from_iterable(s: Sequence[bool | int | float], dtype: Dtype) -> "Storage": ...
    @property
    def size(self) -> int: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def dtype(self) -> Dtype: ...
    def clone(self) -> "Storage": ...
    def cast(self, dtype: Dtype) -> "Storage": ...

class TensorView:
    def __init__(
        self,
        storage: Storage,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
    ) -> None: ...

# Factories
def zeros(shape: tuple[int, ...], dtype: Dtype) -> "Storage": ...
def ones(shape: tuple[int, ...], dtype: Dtype) -> "Storage": ...
def full(
    shape: tuple[int, ...], value: bool | int | float, dtype: Dtype
) -> "Storage": ...
def eye(n: int, dtype: Dtype) -> "Storage": ...
def arange(n: int, start: int, step: int, dtype: Dtype) -> "Storage": ...

# Ops
# s: storage / sh: shape / st: strides / of: offset
def sum(x: TensorView) -> float: ...
def add(x: TensorView, x2: TensorView) -> Storage: ...
def subtract(x: TensorView, x2: TensorView) -> Storage: ...
def multiply(x: TensorView, x2: TensorView) -> Storage: ...
def divide(x: TensorView, x2: TensorView) -> Storage: ...

# Core ops
def cast(s: Storage, dtype: Dtype) -> Storage: ...
def equals(x: TensorView, y: TensorView) -> bool: ...
def copy_view(*, src: TensorView, dst: TensorView) -> Storage: ...
def scatter_to_axes(
    *,
    src: TensorView,
    dst: TensorView,
    fancy_dims_in_src: Sequence[int],
    fancy_dims_data: Sequence[TensorView],
    out_axis_is_fancy: Sequence[bool],
    out_axis_target: Sequence[int],
) -> Storage: ...
def gather_from_axes(
    *,
    x: TensorView,
    new_sh: tuple[int, ...],
    fancy_dims_in_src: Sequence[int],
    fancy_dims_data: Sequence[TensorView],
    out_axis_is_fancy: Sequence[bool],
    out_axis_target: Sequence[int],
) -> Storage: ...
