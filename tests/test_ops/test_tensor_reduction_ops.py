from dataclasses import dataclass
from typing import Any, Callable, Literal

import pytest
import torch

import nanotorch as nt
from nanotorch import testing

nt.manual_seed(42)


def safe_run(fn: Callable) -> tuple[Any | None, Exception | None]:
    """Catch output or exception."""
    try:
        return fn(), None
    except Exception as e:
        return None, e


def make_torch_tensor(x: nt.Tensor | list | float | bool | int) -> torch.Tensor:
    """Create a torch tensor from a nt tensor."""
    if isinstance(x, nt.Tensor):
        x = x.tolist()
    return torch.tensor(x, dtype=torch.float64)


def torch_adapter(
    mode: Literal["min", "max"],
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None,
    keepdim: bool,
):
    """Torch max API aligned with nanotorch."""
    fn = torch.amax if mode == "max" else torch.amin
    if dim is None:
        dim = tuple(range(x.ndim)) if keepdim else ()
    return fn(x, dim=dim, keepdim=keepdim)


@dataclass(kw_only=True)
class ReduceInput:
    """Standardized input spec for reduce operations."""

    x: nt.Tensor | list | float | bool | int
    axis: int | tuple[int, ...] | None = None
    keepdim: bool = False


REDUCE_INPUTS = {
    "scalar": ReduceInput(x=2.0),
    "1d": ReduceInput(x=[0.0, 2.5, -1.5, 8, 7, 3]),
    "1d-ties": ReduceInput(x=[0.0, -1.5, -1.5, 8, 8, 3]),
    "1d-bool": ReduceInput(x=[True, False, True, True, False]),
    "1d-int64": ReduceInput(x=[10, 12, 16, 5, 3, 2]),
    "1d-empty": ReduceInput(x=[]),
    "2d-rand-1": ReduceInput(x=nt.rand(3, 5)),
    "2d-rand-2": ReduceInput(x=nt.rand(5, 5)),
    "2d-rand-3": ReduceInput(x=nt.rand(1, 5)),
    "2d-rand-4": ReduceInput(x=nt.rand(10, 10)),
    "2d-empty": ReduceInput(x=[[], [], []]),
    "3d-rand-1": ReduceInput(x=nt.rand(3, 5, 3)),
    "3d-rand-2": ReduceInput(x=nt.rand(5, 5, 5)),
    "3d-rand-3": ReduceInput(x=nt.rand(1, 5, 5)),
    "3d-rand-4": ReduceInput(x=nt.rand(10, 10, 1)),
    "4d-rand-1": ReduceInput(x=nt.rand(3, 2, 2, 2)),
    "4d-rand-2": ReduceInput(x=nt.rand(5, 2, 3, 2)),
    "4d-rand-3": ReduceInput(x=nt.rand(1, 5, 1, 2)),
    "4d-rand-4": ReduceInput(x=nt.rand(10, 2, 5, 1)),
    "keepdim": ReduceInput(x=nt.rand(10, 2, 5, 1), keepdim=True),
    "empty-axis": ReduceInput(x=nt.rand(10, 2, 5, 1), axis=()),
    "int-axis": ReduceInput(x=nt.rand(10, 2, 5, 1), axis=1),
    "multi-axis": ReduceInput(x=nt.rand(10, 2, 5, 1), axis=(1, 3)),
    "multi-keepdim": ReduceInput(x=nt.rand(10, 2, 5, 1), axis=(1, 3), keepdim=True),
}

REDUCE_OPS = [
    (
        "sum",
        lambda x, **kwargs: x.sum(**kwargs),
        lambda x, **kwargs: x.sum(**kwargs),
    ),
    (
        "mean",
        lambda x, **kwargs: x.mean(**kwargs),
        lambda x, **kwargs: x.mean(**kwargs),
    ),
    (
        "max",
        lambda x, **kwargs: x.max(**kwargs),
        lambda x, **kwargs: torch_adapter("max", x, **kwargs),
    ),
    (
        "min",
        lambda x, **kwargs: x.min(**kwargs),
        lambda x, **kwargs: torch_adapter("min", x, **kwargs),
    ),
    (
        "argmax",
        lambda x, **kwargs: nt.argmax(x, **kwargs),
        lambda x, **kwargs: torch.argmax(x, **kwargs),
    ),
    (
        "argmin",
        lambda x, **kwargs: nt.argmin(x, **kwargs),
        lambda x, **kwargs: torch.argmin(x, **kwargs),
    ),
]

VIEW_FNS = [
    ("contig", lambda t: t),
    ("int-index", lambda t: t[1] if len(t) > 1 and t.shape[0] >= 2 else t),
    ("transposed", lambda t: t.transpose(-1, 0) if t.ndim >= 2 else t),
    ("strided", lambda t: t[..., ::2] if t.ndim and t.shape[-1] >= 2 else t),
    ("fancy", lambda t: t[[0]] if t.ndim and t.shape[0] >= 1 else t),
]


@pytest.mark.parametrize("input_name,input_", REDUCE_INPUTS.items())
@pytest.mark.parametrize("view_name, view_fn", VIEW_FNS)
@pytest.mark.parametrize("op_name, op_nt, op_torch", REDUCE_OPS)
def test_tensor_reduce_value_correct(
    input_name: str,
    view_name: str,
    op_name: str,
    device,
    input_: ReduceInput,
    view_fn: Callable,
    op_nt: Callable,
    op_torch: Callable,
):
    """Tests orchestrator ensuring behavior coherence between nt and torch."""
    if view_name == "fancy" and device == "cuda":
        pytest.skip("CUDA not wired yet for fancy indexing.")
    x_nt = view_fn(
        input_.x
        if isinstance(input_.x, nt.Tensor)
        else nt.tensor(input_.x, dtype=nt.float64, device=device)
    )
    if x_nt is input_.x:
        return
    y_nt, e_nt = safe_run(
        lambda: op_nt(
            x_nt,
            axis=input_.axis,
            keepdim=input_.keepdim,
        )
    )
    x_torch = view_fn(
        input_.x
        if isinstance(input_.x, nt.Tensor)
        else nt.tensor(input_.x, dtype=nt.float64)
    )
    y_torch, e_torch = safe_run(
        lambda: op_torch(
            make_torch_tensor(x_torch),
            dim=input_.axis,
            keepdim=input_.keepdim,
        )
    )
    if e_nt is not None or e_torch is not None:
        assert (e_nt is None) == (e_torch is None), (
            f"divergence: nt={e_nt!r}, torch={e_torch!r}"
        )
        return

    assert y_torch is not None
    assert y_nt is not None
    if device == "cuda":
        y_nt = y_nt.cpu()
    testing.assert_allclose(y_nt, y_torch, tol=1e-4)


# Dtype contracts


SUM_AXIS_KEEPDIM_SPEC = [
    # 1D, axis=0
    ([1.0, 2.0, 3.0], 0, False, (), 6.0),
    ([1.0, 2.0, 3.0], 0, True, (1,), [6.0]),
    ([1.0, 2.0, 3.0], (0,), False, (), 6.0),
    # 1D, axis=None (full reduce)
    ([1.0, 2.0, 3.0], None, False, (), 6.0),
    ([1.0, 2.0, 3.0], None, True, (1,), [6.0]),
    # 2D, axis=0
    ([[1, 2, 3], [4, 5, 6]], 0, False, (3,), [5, 7, 9]),
    ([[1, 2, 3], [4, 5, 6]], 0, True, (1, 3), [[5, 7, 9]]),
    # 2D, axis=1
    ([[1, 2, 3], [4, 5, 6]], 1, False, (2,), [6, 15]),
    ([[1, 2, 3], [4, 5, 6]], 1, True, (2, 1), [[6], [15]]),
    # 2D, negative axis
    ([[1, 2, 3], [4, 5, 6]], -1, False, (2,), [6, 15]),
    ([[1, 2, 3], [4, 5, 6]], -2, False, (3,), [5, 7, 9]),
    # 2D, axis=None
    ([[1, 2, 3], [4, 5, 6]], None, False, (), 21.0),
    ([[1, 2, 3], [4, 5, 6]], None, True, (1, 1), [[21.0]]),
    # 2D, multi-axis (full)
    ([[1, 2, 3], [4, 5, 6]], (0, 1), False, (), 21.0),
    ([[1, 2, 3], [4, 5, 6]], (0, 1), True, (1, 1), [[21.0]]),
    # 3D
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 0, False, (2, 2), [[6, 8], [10, 12]]),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 1, False, (2, 2), [[4, 6], [12, 14]]),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, False, (2, 2), [[3, 7], [11, 15]]),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (0, 2), False, (2,), [14, 22]),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (0, 2), True, (1, 2, 1), [[[14], [22]]]),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], None, False, (), 36.0),
]


@pytest.mark.parametrize(
    "input,axis,keepdim,expected_shape,expected", SUM_AXIS_KEEPDIM_SPEC
)
def test_tensor_sum_axis(device, input, axis, keepdim, expected_shape, expected):
    x = nt.tensor(input).to(device)
    out = x.sum(axis=axis, keepdim=keepdim)
    if device == "cuda":
        out = out.cpu()
    assert out.shape == expected_shape
    assert out.tolist() == expected


@pytest.mark.parametrize(
    "input,in_dtype,out_dtype",
    [
        ([True, False, True], nt.bool_, nt.int64),
        ([1, 2, 3], nt.int32, nt.int64),
        ([1, 2, 3], nt.int64, nt.int64),
        ([1.0, 2.0, 3.0], nt.float32, nt.float32),
        ([1.0, 2.0, 3.0], nt.float64, nt.float64),
    ],
)
def test_tensor_sum_default_dtype_promotion(input, in_dtype, out_dtype):
    x = nt.tensor(input, dtype=in_dtype)
    assert x.sum().dtype == out_dtype
    assert x.sum(axis=0).dtype == out_dtype


@pytest.mark.parametrize(
    "input,in_dtype,kwarg_dtype,expected",
    [
        ([1, 2, 3], nt.int32, nt.float32, 6.0),
        ([1, 2, 3], nt.int32, nt.float64, 6.0),
        ([1.5, 2.5], nt.float32, nt.float64, 4.0),
        ([True, True, False], nt.bool_, nt.int32, 2),
        ([True, True, False], nt.bool_, nt.float32, 2.0),
    ],
)
def test_tensor_sum_dtype_kwarg(input, in_dtype, kwarg_dtype, expected):
    x = nt.tensor(input, dtype=in_dtype)
    out = x.sum(dtype=kwarg_dtype)
    assert out.dtype == kwarg_dtype
    assert out.tolist() == expected


def test_tensor_sum_dtype_kwarg_with_axis_and_keepdim():
    x = nt.tensor([[1, 2], [3, 4]], dtype=nt.int32)
    out = x.sum(axis=0, keepdim=True, dtype=nt.float64)
    assert out.shape == (1, 2)
    assert out.dtype == nt.float64
    assert out.tolist() == [[4.0, 6.0]]


@pytest.mark.parametrize(
    "in_dtype,kwarg_dtype,expected",
    [
        (nt.int32, nt.float32, 2.0),
        (nt.int64, nt.float64, 2.0),
        (nt.float32, nt.float64, 2.0),
    ],
)
def test_tensor_mean_dtype_kwarg(in_dtype, kwarg_dtype, expected):
    x = nt.tensor([1, 2, 3], dtype=in_dtype)
    out = x.mean(dtype=kwarg_dtype)
    assert out.dtype == kwarg_dtype
    assert out.tolist() == expected


def test_tensor_mean_non_float_dtype_kwarg_raises():
    x = nt.tensor([1.0, 2.0, 3.0])
    with pytest.raises((TypeError, ValueError)):
        x.mean(dtype=nt.int32)


# Issues


def test_tensor_sum_backward_non_leading_axis():  # See #37
    x = nt.arange(60, requires_grad=True).reshape(3, 4, 5)
    y = x.sum(axis=1).sum()
    y.backward()
