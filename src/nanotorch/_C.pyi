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

# s: storage / sh: shape / st: strides / of: offset
def cast(s: Storage, dtype: Dtype) -> Storage: ...
def equals(
    s1: Storage,
    sh1: tuple,
    st1: tuple,
    of1: int,
    s2: Storage,
    sh2: tuple,
    st2: tuple,
    of2: int,
) -> bool: ...
def sum(s: Storage, sh: tuple, st: tuple, of: int) -> float: ...
