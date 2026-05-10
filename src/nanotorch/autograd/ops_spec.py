"""Fundamental components to define an op specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
