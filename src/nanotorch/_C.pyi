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
def cast(s: Storage, dtype: Dtype) -> Storage: ...
def equals(
    s1: Storage,
    sh1: tuple[int, ...],
    st1: tuple[int, ...],
    of1: int,
    s2: Storage,
    sh2: tuple[int, ...],
    st2: tuple[int, ...],
    of2: int,
) -> bool: ...
def sum(s: Storage, sh: tuple[int, ...], st: tuple[int, ...], of: int) -> float: ...
