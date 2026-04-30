"""Main interface for C++ kernel symbols."""

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

class Storage(Buffer):
    """Pointer to a 1D contiguous, typed memory segment."""
    @staticmethod
    def allocate(n: int, dtype: Dtype) -> "Storage":
        """Allocate a new storage of given size (# of elements) and dtype."""
    @staticmethod
    def from_iterable(s: Sequence[bool | int | float], dtype: Dtype) -> "Storage":
        """Allocate a new storage from a sequence of python elements."""
    @property
    def size(self) -> int:
        """Number of elements in the storage."""
    @property
    def itemsize(self) -> int:
        """Memory size of one storage element, in bits."""
    @property
    def dtype(self) -> Dtype:
        """Type of elements in the storage."""
    def clone(self) -> "Storage":
        """Clone the storage."""
    def cast(self, dtype: Dtype) -> "Storage":
        """Cast the storage to a new dtype, returns a new storage if type changes."""

class TensorView:
    """C++ representation of a strided tensor."""
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
def sum(x: TensorView, axis: Sequence[int], keepdim: bool, dtype: Dtype) -> Storage: ...
def add(x1: TensorView, x2: TensorView) -> Storage: ...
def subtract(x1: TensorView, x2: TensorView) -> Storage: ...
def multiply(x1: TensorView, x2: TensorView) -> Storage: ...
def divide(x1: TensorView, x2: TensorView) -> Storage: ...
def matmul(x1: TensorView, x2: TensorView) -> Storage: ...
def exp(x1: TensorView) -> Storage: ...
def log(x1: TensorView) -> Storage: ...
def pow(x1: TensorView, power: float) -> Storage: ...

# Special
def relu(x: TensorView) -> Storage: ...
def greater(x: TensorView, s: float | int | bool) -> Storage: ...

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
