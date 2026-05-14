"""Fundamental components to define an op specification."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from nanotorch import _data_type as dt
from nanotorch.core import Tensor, TensorLike
from nanotorch.factories import rand, randint


@dataclass(frozen=True)
class Input:
    """Defines the specification of one input in a Tensor operation.

    Parameters
    ----------
    kind: "tensor" or "scalar"
        Type of input (scalar or tensor, 1-numel tensor counts as both).
    shape: InputShape
        Input shape, can be defined relative to other inputs.
    domain: InputDomain
        Input number domain.
    """

    kind: Literal["tensor", "scalar"]
    shape: InputShape
    domain: InputDomain


@dataclass
class AnyShape:
    min_ndim: int = 0


@dataclass
class BroadcastableTo:
    ref: int  # -> ref input position


@dataclass
class MatmulBroadcast:
    ref: int  # -> ref input position


class Bool: ...


@dataclass
class Real:
    low: float = -500
    high: float = 500


@dataclass
class Axis:
    ref: int


@dataclass
class AxisSet:
    ref: int
    min: int
    split: bool


@dataclass
class AxisPermutation:
    ref: int


@dataclass
class AxisReshape:
    ref: int


@dataclass
class AxisExpand:
    ref: int


InputShape = AnyShape | BroadcastableTo | MatmulBroadcast | None
InputDomain = Bool | Real | Axis | AxisSet | AxisPermutation | AxisReshape | AxisExpand


def gen_random_input_for(
    op_spec: tuple[Input, ...], *, min_ndim: int, max_ndim: int, size_factor: int
) -> list[TensorLike]:
    """Generate a valid op input based on the spec."""
    inputs = []
    for input_ in op_spec:
        domain = input_.domain
        if isinstance(domain, (Bool, Real)):
            inputs.append(
                _gen_float_like_input(input_, inputs, min_ndim, max_ndim, size_factor)
            )
        else:
            inputs.extend(_gen_axis_like_input(domain, inputs))
    return inputs


def _gen_float_like_input(
    input_: Input,
    inputs: list[TensorLike],
    min_ndim: int,
    max_ndim: int,
    size_factor: int,
) -> TensorLike:
    """Generate a numeric tensor input based on the spec."""
    if input_.kind == "scalar":
        assert input_.shape is None
        shape = ()
    elif isinstance(input_.shape, AnyShape):
        ndim = random.randint(max(min_ndim, input_.shape.min_ndim), max_ndim)
        shape = randint(1 * size_factor, 8 * size_factor, (ndim,)).tolist()
    elif isinstance(input_.shape, BroadcastableTo):
        ref = inputs[input_.shape.ref]
        assert isinstance(ref, Tensor)
        ndim = random.randint(0, ref.ndim + 1)
        suffix = ref.shape[ref.ndim - ndim :]
        shape = [1 if random.random() < 0.5 else ax for ax in suffix]
    elif isinstance(input_.shape, MatmulBroadcast):
        ref = inputs[input_.shape.ref]
        assert isinstance(ref, Tensor)
        shape = [1 if random.random() < 0.5 else ax for ax in ref.shape[:-2]]
        shape += [ref.shape[-1], random.randint(1, 8)]
    else:
        raise AssertionError(f"Unhandled shape spec: {input_.shape}")
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, list):
        shape = tuple(shape)
    assert isinstance(shape, tuple)

    if isinstance(input_.domain, Real):
        dtype = dt.float32
    elif isinstance(input_.domain, Bool):
        dtype = dt.bool_
    else:
        raise ValueError(f"Unrecognized type {input_.domain}")
    x = rand(*shape, dtype=dt.float64, requires_grad=False).to(dtype)

    if isinstance(input_.domain, Real):
        x = (input_.domain.high - input_.domain.low) * x + input_.domain.low
    if input_.kind == "tensor":
        x = x.clone()
        x.enable_grad()
    else:
        x = x.item()
    return x


def _gen_axis_like_input(
    domain: InputDomain, inputs: list[TensorLike]
) -> tuple[int, ...] | list[tuple[int, ...]]:
    """Generate an axis-derived input based on the spec."""
    ref = inputs[domain.ref]  # type: ignore
    assert isinstance(ref, Tensor)
    if isinstance(domain, (Axis, AxisSet)):
        naxes = (
            1 if isinstance(domain, (Axis)) else random.randint(domain.min, ref.ndim)
        )
        axes = tuple(random.sample(range(ref.ndim), naxes))
        if isinstance(domain, AxisSet) and not domain.split:
            return [axes]
        return axes
    elif isinstance(domain, AxisPermutation):
        return tuple(random.sample(range(ref.ndim), ref.ndim))
    elif isinstance(domain, AxisReshape):
        return _gen_reshape(ref)
    elif isinstance(domain, AxisExpand):
        return _gen_expand(ref)
    else:
        raise AssertionError(f"Cannot handle domain {domain}")


def _gen_reshape(ref: Tensor) -> tuple[int, ...]:
    factors, n = [], math.prod(ref.shape)
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1
    if n > 1:
        factors.append(n)
    ndim: int = 1 + random.randint(0, len(factors))
    buckets = [1] * ndim
    for f in factors:
        buckets[random.randint(0, len(buckets) - 1)] *= f
    buckets += [1] * random.randint(0, 5)
    random.shuffle(buckets)
    return tuple(buckets)  # type: ignore


def _gen_expand(ref: Tensor) -> tuple[int, ...]:
    shape = []
    for ax in ref.shape[::-1]:
        shape = [random.randint(1, 8) if ax == 1 else ax] + shape
    while True:
        shape = [random.randint(1, 8)] + shape
        if random.random() < 0.5:
            break
    return tuple(shape)
