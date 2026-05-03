"""Usual tensor factories."""

import math

from nanotorch import _C

from . import _data_type as dt
from ._device import Device, DeviceLiteral, get_std_device
from .core import Dtype, InputType, Tensor, inherit_doc


@inherit_doc(Tensor)
def tensor(
    data: InputType,
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    device: Device = Device.Cpu,
) -> Tensor:
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, device=device)


def zeros(
    *shape: int,
    dtype: Dtype = dt.float32,
    requires_grad: bool = False,
    device: Device | DeviceLiteral = Device.Cpu,
) -> Tensor:
    """Initialize a new tensor filled with zeros.

    Parameters
    ----------
    *shape: int
        Dimensions of the tensor.
    dtype: Dtype
        Data type to cast the values to. If not set, the tensor will be promoted
        to the most precise data type following the DataTypes enum order.
    requires_grad: bool
        Enables autograd for this tensor, and all tensors constructed from it
        through differentiable operations.
    device: Device | "cpu" | "cuda"
        Device to store the Tensor on. Operations will be automatically performed
        on device if there is an available kernel.
    """
    return _tensor_factory(
        *shape,
        storage=_C.zeros(math.prod(shape), dtype, get_std_device(device)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def ones(
    *shape: int,
    dtype: Dtype = dt.float32,
    requires_grad: bool = False,
    device: Device | DeviceLiteral = Device.Cpu,
) -> Tensor:
    """Initialize a new tensor filled with ones.

    Parameters
    ----------
    *shape: int
        Dimensions of the tensor.
    dtype: Dtype
        Data type to cast the values to. If not set, the tensor will be promoted
        to the most precise data type following the DataTypes enum order.
    requires_grad: bool
        Enables autograd for this tensor, and all tensors constructed from it
        through differentiable operations.
    device: Device | "cpu" | "cuda"
        Device to store the Tensor on. Operations will be automatically performed
        on device if there is an available kernel.
    """
    return _tensor_factory(
        *shape,
        storage=_C.ones(math.prod(shape), dtype, get_std_device(device)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def full(
    *shape: int,
    value: bool | int | float,
    dtype: Dtype = dt.float32,
    requires_grad: bool = False,
    device: Device | DeviceLiteral = Device.Cpu,
) -> Tensor:
    """Initialize a new tensor filled with a set value.

    Parameters
    ----------
    *shape: int
        Dimensions of the tensor.
    value: bool | int | float
        Value to fill the tensor with.
    dtype: Dtype
        Data type to cast the values to. If not set, the tensor will be promoted
        to the most precise data type following the DataTypes enum order.
    requires_grad: bool
        Enables autograd for this tensor, and all tensors constructed from it
        through differentiable operations.
    device: Device | "cpu" | "cuda"
        Device to store the Tensor on. Operations will be automatically performed
        on device if there is an available kernel.
    """
    return _tensor_factory(
        *shape,
        storage=_C.full(math.prod(shape), value, dtype, get_std_device(device)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def eye(
    n: int,
    dtype: Dtype = dt.float32,
    requires_grad: bool = False,
    device: Device | DeviceLiteral = Device.Cpu,
) -> Tensor:
    """Initialize a new eye matrix of rank n.

    Parameters
    ----------
    n: int
        Matrix rank, output will be (n, n).
    dtype: Dtype
        Data type to cast the values to. If not set, the tensor will be promoted
        to the most precise data type following the DataTypes enum order.
    requires_grad: bool
        Enables autograd for this tensor, and all tensors constructed from it
        through differentiable operations.
    device: Device | "cpu" | "cuda"
        Device to store the Tensor on. Operations will be automatically performed
        on device if there is an available kernel.
    """
    return _tensor_factory(
        *(n, n),
        storage=_C.eye(n, dtype, get_std_device(device)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def arange(
    n: int,
    start: int = 0,
    step: int = 1,
    dtype: Dtype = dt.int64,
    requires_grad: bool = False,
    device: Device | DeviceLiteral = Device.Cpu,
) -> Tensor:
    """Initialize a new arithmetic range.

    If `x = arange(n, a0, r)`, `len(x) = n` and `x[i] = a0 + i * r`.

    Parameters
    ----------
    n: int
        Number of elements in the range.
    start: int
        First element in the range.
    step: int
        Increment of the range.
    dtype: Dtype
        Data type to cast the values to. If not set, the tensor will be promoted
        to the most precise data type following the DataTypes enum order.
    requires_grad: bool
        Enables autograd for this tensor, and all tensors constructed from it
        through differentiable operations.
    device: Device | "cpu" | "cuda"
        Device to store the Tensor on. Operations will be automatically performed
        on device if there is an available kernel.
    """
    return _tensor_factory(
        n,
        storage=_C.arange(n, start, step, dtype, get_std_device(device)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def rand(
    *shape: int,
    dtype: Dtype = dt.float32,
    requires_grad: bool = False,
    device: Device | DeviceLiteral = Device.Cpu,
) -> Tensor:
    """Initialize a new tensor initialized with random noise.

    Parameters
    ----------
    *shape: int
        Dimensions of the tensor.
    dtype: Dtype
        Data type to cast the values to. If not set, the tensor will be promoted
        to the most precise data type following the DataTypes enum order.
    requires_grad: bool
        Enables autograd for this tensor, and all tensors constructed from it
        through differentiable operations.
    device: Device | "cpu" | "cuda"
        Device to store the Tensor on. Operations will be automatically performed
        on device if there is an available kernel.
    """
    from numpy.random import rand

    return Tensor(rand(*shape), dtype=dtype, requires_grad=requires_grad, device=device)  # type: ignore


def _tensor_factory(
    *shape: int,
    storage: _C.Storage,
    dtype: Dtype = dt.float32,
    requires_grad: bool = False,
) -> Tensor:
    """Generic tensor factory."""
    if isinstance(shape, int):
        shape = (shape,)
    x = Tensor._new_contiguous(storage, shape)
    if requires_grad:
        x.enable_grad()
    return x
