import pytest
import torch
from conftest import requires_cuda

import nanotorch as nt
from nanotorch import testing

nt.manual_seed(42)

# TODO: add empty checks

REDUCE_INPUTS = {
    "scalar": 2.0,
    "1d": [0.0, 2.5, -1.5, 8, 7, 3],
    "1d-ties": [0.0, -1.5, -1.5, 8, 8, 3],
    "2d-rand-1": nt.rand(3, 5).tolist(),
    "2d-rand-2": nt.rand(5, 5).tolist(),
    "2d-rand-3": nt.rand(1, 5).tolist(),
    "2d-rand-4": nt.rand(10, 10).tolist(),
    "3d-rand-1": nt.rand(3, 5, 3).tolist(),
    "3d-rand-2": nt.rand(5, 5, 5).tolist(),
    "3d-rand-3": nt.rand(1, 5, 5).tolist(),
    "3d-rand-4": nt.rand(10, 10, 1).tolist(),
    "4d-rand-1": nt.rand(3, 2, 2, 2).tolist(),
    "4d-rand-2": nt.rand(5, 2, 3, 2).tolist(),
    "4d-rand-3": nt.rand(1, 5, 1, 2).tolist(),
    "4d-rand-4": nt.rand(10, 2, 5, 1).tolist(),
}


@pytest.mark.parametrize(
    "name_op, op_nt, op_torch",
    [
        ("sum", lambda x: x.sum(), lambda x: x.sum()),
        ("max", lambda x: x.max(), lambda x: x.max()),
        ("min", lambda x: x.min(), lambda x: x.min()),
        ("argmin", lambda x: nt.argmin(x), lambda x: torch.argmin(x)),
        ("argmax", lambda x: nt.argmax(x), lambda x: torch.argmax(x)),
    ],
)
@pytest.mark.parametrize("name,input", REDUCE_INPUTS.items())
def test_tensor_reduce(name, op_nt, op_torch, name_op, input):
    y_nt = op_nt(nt.tensor(input, dtype=nt.float64))
    y_torch = op_torch(torch.tensor(input, dtype=torch.float64))
    testing.assert_allclose(y_nt, y_torch, tol=1e-4)


@requires_cuda
@pytest.mark.parametrize(
    "name_op, op_nt, op_torch",
    [
        ("sum", lambda x: x.sum(), lambda x: x.sum()),
        ("max", lambda x: x.max(), lambda x: x.max()),
        ("min", lambda x: x.min(), lambda x: x.min()),
        ("argmin", lambda x: nt.argmin(x), lambda x: torch.argmin(x)),
        ("argmax", lambda x: nt.argmax(x), lambda x: torch.argmax(x)),
    ],
)
@pytest.mark.parametrize("name,input", REDUCE_INPUTS.items())
def test_tensor_reduce_cuda(name, op_nt, op_torch, name_op, input):
    y_nt = op_nt(nt.tensor(input, dtype=nt.float64, device="cuda")).cpu()
    y_torch = op_torch(torch.tensor(input, dtype=torch.float64))
    testing.assert_allclose(y_nt, y_torch, tol=1e-4)


def test_tensor_sum_empty_is_all():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert x.sum(axis=()).tolist() == 45


def test_tensor_sum_transpose():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert x.sum().tolist() == x.T.sum().tolist()


def test_tensor_sum_1d_intindex_slice():
    x = nt.tensor([1, 2, 3, 4, 5, 6])
    assert x[3].sum().tolist() == 4


def test_tensor_sum_2d_intindex_slice():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert x[1].sum().tolist() == 15


def test_tensor_sum_backward_non_leading_axis():  # See #37
    x = nt.arange(60, requires_grad=True).reshape(3, 4, 5)
    y = x.sum(axis=1).sum()
    y.backward()


# Sum with axis / keepdim

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
def test_tensor_sum_axis(input, axis, keepdim, expected_shape, expected):
    x = nt.tensor(input)
    out = x.sum(axis=axis, keepdim=keepdim)
    assert out.shape == expected_shape
    assert out.tolist() == expected


@requires_cuda
@pytest.mark.parametrize(
    "input,axis,keepdim,expected_shape,expected", SUM_AXIS_KEEPDIM_SPEC
)
def test_tensor_sum_axis_cuda(input, axis, keepdim, expected_shape, expected):
    x = nt.tensor(input)
    out = x.to("cuda").sum(axis=axis, keepdim=keepdim)
    assert out.device == nt.Device.Cuda
    assert out.shape == expected_shape
    assert out.cpu().tolist() == expected


# Sum on views


def test_tensor_sum_axis_transposed():
    x = nt.tensor([[1, 2, 3], [4, 5, 6]])
    assert x.T.sum(axis=0).tolist() == x.sum(axis=1).tolist()
    assert x.T.sum(axis=1).tolist() == x.sum(axis=0).tolist()


def test_tensor_sum_axis_sliced_view():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    v = x[:, ::2]
    assert v.tolist() == [[1, 3], [5, 7], [9, 11]]
    assert v.sum(axis=0).tolist() == [15, 21]
    assert v.sum(axis=1).tolist() == [4, 12, 20]


def test_tensor_sum_axis_fancy_indexed():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    picked = x[[0, 2]]
    assert picked.sum(axis=0).tolist() == [8, 10, 12]
    assert picked.sum(axis=1).tolist() == [6, 24]


@requires_cuda
def test_tensor_sum_axis_transposed_cuda():
    x = nt.tensor([[1, 2, 3], [4, 5, 6]]).to("cuda")
    assert x.T.sum(axis=0).cpu().tolist() == x.sum(axis=1).cpu().tolist()
    assert x.T.sum(axis=1).cpu().tolist() == x.sum(axis=0).cpu().tolist()


@requires_cuda
def test_tensor_sum_axis_sliced_view_cuda():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    v = x[:, ::2].to("cuda")
    assert v.cpu().tolist() == [[1, 3], [5, 7], [9, 11]]
    assert v.sum(axis=0).cpu().tolist() == [15, 21]
    assert v.sum(axis=1).cpu().tolist() == [4, 12, 20]


@requires_cuda
def test_tensor_sum_axis_fancy_indexed_cuda():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    picked = x[[0, 2]].to("cuda")
    assert picked.sum(axis=0).cpu().tolist() == [8, 10, 12]
    assert picked.sum(axis=1).cpu().tolist() == [6, 24]


# Sum across dtypes (values only)

SUM_DTYPE_SPEC = [
    ([[True, False], [True, True]], nt.bool_, 0, [2, 1]),
    ([[True, False], [True, True]], nt.bool_, 1, [1, 2]),
    ([[1, 2], [3, 4]], nt.int32, 0, [4, 6]),
    ([[1, 2], [3, 4]], nt.int64, 1, [3, 7]),
    ([[1.5, 2.5], [3.0, 4.0]], nt.float32, 0, [4.5, 6.5]),
    ([[1.5, 2.5], [3.0, 4.0]], nt.float64, 1, [4.0, 7.0]),
]


@pytest.mark.parametrize("input,dtype,axis,expected", SUM_DTYPE_SPEC)
def test_tensor_sum_axis_dtypes(input, dtype, axis, expected):
    x = nt.tensor(input, dtype=dtype)
    assert x.sum(axis=axis).tolist() == expected


@requires_cuda
@pytest.mark.parametrize("input,dtype,axis,expected", SUM_DTYPE_SPEC)
def test_tensor_sum_axis_dtypes_cuda(input, dtype, axis, expected):
    x = nt.tensor(input, dtype=dtype)
    out = x.to("cuda").sum(axis=axis)
    assert out.device == nt.Device.Cuda
    assert out.cpu().tolist() == expected


# Sum dtype contract (torch auto-promote: bool/int -> int64, float unchanged)


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


# Sum dtype= kwarg: explicit override, output matches requested dtype


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


# Sum corner cases (torch conventions)


def test_tensor_sum_axis_empty_1d():
    x = nt.tensor([])
    assert x.sum(axis=0).shape == ()
    assert x.sum(axis=0).tolist() == 0


def test_tensor_sum_axis_zero_dim_2d_reduce_empty_axis():
    x = nt.tensor([[], [], []])
    out = x.sum(axis=1)
    assert out.shape == (3,)
    assert out.tolist() == [0, 0, 0]


def test_tensor_sum_axis_zero_dim_2d_reduce_kept_axis():
    x = nt.tensor([[], [], []])
    out = x.sum(axis=0)
    assert out.shape == (0,)
    assert out.tolist() == []


def test_tensor_sum_keepdim_preserves_rank():
    x = nt.tensor([[1, 2, 3], [4, 5, 6]])
    out = x.sum(axis=1, keepdim=True)
    assert out.shape == (2, 1)
    assert (x - out).shape == x.shape


def test_tensor_sum_scalar_no_axis():
    x = nt.tensor(5.0)
    assert x.sum().tolist() == 5.0


# Sum invalid input (torch conventions -> IndexError)


def test_tensor_sum_axis_out_of_range_positive():
    x = nt.tensor([[1, 2], [3, 4]])
    with pytest.raises(IndexError):
        x.sum(axis=2)


def test_tensor_sum_axis_out_of_range_negative():
    x = nt.tensor([[1, 2], [3, 4]])
    with pytest.raises(IndexError):
        x.sum(axis=-3)


def test_tensor_sum_axis_duplicate():
    x = nt.tensor([[1, 2], [3, 4]])
    with pytest.raises((IndexError, ValueError)):
        x.sum(axis=(0, 0))


def test_tensor_sum_axis_on_scalar_raises():
    x = nt.tensor(5.0)
    with pytest.raises(IndexError):
        x.sum(axis=0)


# Mean


@pytest.mark.parametrize(
    "input,axis,keepdim,expected_shape,expected",
    [
        ([1.0, 2.0, 3.0, 4.0], None, False, (), 2.5),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 0, False, (3,), [2.5, 3.5, 4.5]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 1, False, (2,), [2.0, 5.0]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 1, True, (2, 1), [[2.0], [5.0]]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], None, False, (), 3.5),
        (
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            (0, 2),
            False,
            (2,),
            [3.5, 5.5],
        ),
    ],
)
def test_tensor_mean_axis(input, axis, keepdim, expected_shape, expected):
    x = nt.tensor(input)
    out = x.mean(axis=axis, keepdim=keepdim)
    assert out.shape == expected_shape
    assert out.tolist() == expected


def test_tensor_mean_transposed():
    x = nt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert x.T.mean(axis=0).tolist() == x.mean(axis=1).tolist()


# Mean dtype contract


@pytest.mark.parametrize(
    "in_dtype",
    [nt.bool_, nt.int32, nt.int64],
)
def test_tensor_mean_int_input_raises(in_dtype):
    x = nt.tensor([1, 2, 3], dtype=in_dtype)
    with pytest.raises((TypeError, RuntimeError)):
        x.mean()


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


def test_tensor_mean_backward_non_leading_axis():  # See #37
    x = nt.arange(60, dtype=nt.float32, requires_grad=True).reshape(3, 4, 5)
    y = x.mean(axis=1).sum()
    y.backward()
