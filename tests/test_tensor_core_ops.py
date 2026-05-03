import math

import numpy as np
import pytest

import nanotorch as nt
from nanotorch import testing

# Tolist


def test_tolist():
    x = nt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert x.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_tolist_extensive():
    np.random.seed(42)
    for _ in range(100):
        ndim = np.random.randint(0, 5)
        shape = np.random.randint(0, 5, (ndim,))
        xt = np.random.randint(-10, 10, shape).tolist()
        assert nt.tensor(xt).tolist() == xt


# Sum


@pytest.mark.parametrize(
    "input,expected",
    [
        (([],), 0.0),
        ((2.0,), 2.0),
        (([1.0, 2.0, 3.0],), 6.0),
        (([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), 21.0),
        (([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],), 36.0),
        (([1.0, 2.0, -3.0],), 0.0),
        (([-1.0, -2.0, -3.0],), -6.0),
        (([True, True, False, True],), 3),
        (([1, 7, 0, 8], nt.int32), 16),
        (([1, 7, 0, 8], nt.int64), 16),
        (([1.5, 7.5, -1.5, 8.0], nt.float32), 15.5),
        (([1.5, 7.5, -1.5, 8.0], nt.float64), 15.5),
    ],
)
def test_tensor_sum(input, expected):
    if len(input) == 2:
        input, dtype = input
        x = nt.tensor(input, dtype=dtype)
    else:
        x = nt.tensor(*input)
    result = x.sum()
    assert result.tolist() == expected


def test_tensor_sum_transpose():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert x.sum().tolist() == x.T.sum().tolist()


def test_tensor_sum_1d_intindex_slice():
    x = nt.tensor([1, 2, 3, 4, 5, 6])
    assert x[3].sum().tolist() == 4


def test_tensor_sum_2d_intindex_slice():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert x[1].sum().tolist() == 15


# Sum with axis / keepdim


@pytest.mark.parametrize(
    "input,axis,keepdim,expected_shape,expected",
    [
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
    ],
)
def test_tensor_sum_axis(input, axis, keepdim, expected_shape, expected):
    x = nt.tensor(input)
    out = x.sum(axis=axis, keepdim=keepdim)
    assert out.shape == expected_shape
    assert out.tolist() == expected


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


# Sum across dtypes (values only)


@pytest.mark.parametrize(
    "input,dtype,axis,expected",
    [
        ([[True, False], [True, True]], nt.bool_, 0, [2, 1]),
        ([[True, False], [True, True]], nt.bool_, 1, [1, 2]),
        ([[1, 2], [3, 4]], nt.int32, 0, [4, 6]),
        ([[1, 2], [3, 4]], nt.int64, 1, [3, 7]),
        ([[1.5, 2.5], [3.0, 4.0]], nt.float32, 0, [4.5, 6.5]),
        ([[1.5, 2.5], [3.0, 4.0]], nt.float64, 1, [4.0, 7.0]),
    ],
)
def test_tensor_sum_axis_dtypes(input, dtype, axis, expected):
    x = nt.tensor(input, dtype=dtype)
    assert x.sum(axis=axis).tolist() == expected


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


# Equals


def test_tensor_equals_shape():
    x = nt.tensor([1, 2, 3])
    y = nt.tensor([1, 2])
    assert not x.equals(y)


def test_tensor_equals_type():
    i32 = nt.tensor([1, 2, 3], dtype=nt.int32)
    i64 = nt.tensor([1, 2, 3], dtype=nt.int64)
    f32 = nt.tensor([1, 2, 3], dtype=nt.float32)
    f64 = nt.tensor([1, 2, 3], dtype=nt.float64)
    assert i32.equals(i64)
    assert i32.equals(f32)
    assert i32.equals(f64)


def test_tensor_equals_empty():
    assert nt.tensor([]).equals(nt.tensor([]))


def test_tensor_unequal():
    x = nt.tensor([1, 2, 3, 4, 5])
    y = nt.tensor([1, 2, 2, 4, 5])
    assert not x.equals(y)


def test_tensor_unequal_close():
    x = nt.tensor([1, 2, 3, 4, 5])
    y = nt.tensor([1, 2, 3.00001, 4, 5])
    assert not x.equals(y)


def test_tensor_equals_1d_intindex_slice():
    x = nt.tensor([1, 2, 3, 4, 5, 6])
    xt = [1, 2, 3, 4, 5, 6]
    for i in range(6):
        assert x[i].equals(nt.tensor(xt[i]))


def test_tensor_equals_2d_intindex_slice():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    xt0 = nt.tensor([1, 2, 3])
    xt1 = nt.tensor([4, 5, 6])
    xt2 = nt.tensor([7, 8, 9])
    assert x[0].equals(xt0)
    assert x[1].equals(xt1)
    assert x[2].equals(xt2)


# Transpose


def test_tensor_transpose_empty():
    x = nt.tensor([])
    assert x.T.dtype == x.dtype
    assert x.T.shape == x.shape
    assert x.T.tolist() == x.tolist()


def test_tensor_transpose_0d():
    x = nt.tensor(0.0)
    assert x.T.dtype == x.dtype
    assert x.T.shape == x.shape
    assert x.T.tolist() == x.tolist()


def test_tensor_transpose_1d():
    x = nt.tensor([0.0, 1.0, 2.0, 3.0])
    assert x.T.dtype == x.dtype
    assert x.T.shape == x.shape
    assert x.T.tolist() == x.tolist()


def test_tensor_transpose_2d():
    x = nt.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]
    )
    assert x.T.tolist() == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
    ]


def test_tensor_transpose_3d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert x.T.tolist() == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]


def test_cast_after_transpose():
    x = nt.tensor([[1, 2, 3], [4, 5, 6]])
    assert x.T.to(nt.float64).tolist() == [[1, 4], [2, 5], [3, 6]]


def test_manual_transpose():
    x = nt.tensor([[1, 2, 3], [4, 5, 6]])
    assert x.T.equals(x.transpose(0, 1))


# Contiguous


def test_tensor_contiguous():
    x = nt.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]
    )
    assert x._is_contiguous()


def test_tensor_not_contiguous():
    x = nt.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]
    )
    assert not x.T._is_contiguous()


def test_tensor_to_contiguous():
    x = nt.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]
    )
    assert x._to_contiguous() is x


def test_tensor_not_contiguous_to_contiguous():
    x = nt.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]
    )
    assert x.T._to_contiguous()._is_contiguous()


# Reshape


@pytest.mark.parametrize(
    "input,shape,expected",
    [
        (0.0, (1,), [0.0]),
        ([1, 2, 3, 4, 5, 6], (2, 3), [[1, 2, 3], [4, 5, 6]]),
        ([1, 2, 3, 4, 5, 6, 7, 8], (2, 2, 2), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        ([1, 2, 3, 4, 5, 6], (1, 6), [[1, 2, 3, 4, 5, 6]]),
        ([[1, 2, 3], [4, 5, 6]], (6,), [1, 2, 3, 4, 5, 6]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (8,), [1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_reshape(input, shape, expected):
    x = nt.tensor(input)
    assert x.reshape(*shape).tolist() == expected


def test_reshape_empty():
    x = nt.tensor([1, 2, 3])
    with pytest.raises(ValueError):
        x.reshape()


def test_reshape_1d_identity():
    x = nt.tensor([1, 2, 3])
    assert x.reshape(3) is x


def test_reshape_no_contiguous():
    x = nt.tensor([[1, 2, 3], [4, 5, 6]])
    assert x.T.reshape(6).tolist() == [1, 4, 2, 5, 3, 6]


# Weird cases


def test_zero_dim_shape_2d():
    x = nt.tensor([[], [], []])
    assert x.shape == (3, 0)
    assert x.tolist() == [[], [], []]
    assert x.sum().tolist() == 0


def test_zero_dim_shape_3d_middle():
    x = nt.tensor([[[]], [[]]])
    assert x.shape == (2, 1, 0)
    assert x.tolist() == [[[]], [[]]]
    assert x.sum().tolist() == 0


# item()


@pytest.mark.parametrize(
    "input,expected",
    [
        (0, 0),
        (1.5, 1.5),
        (True, True),
        (False, False),
        ([0], 0),
        ([3.5], 3.5),
        ([[5]], 5),
        ([[[[True]]]], True),
    ],
)
def test_item_valid(input, expected):
    assert nt.tensor(input).item() == expected


def test_item_empty_raises():
    with pytest.raises(RuntimeError):
        nt.tensor([]).item()


def test_item_toomany_raises():
    with pytest.raises(RuntimeError):
        nt.tensor([1, 2]).item()


# exp/log/pow


@pytest.mark.parametrize(
    "input,expected",
    [
        (0.0, 1.0),
        (1.0, math.e),
        ([0, 1], [1.0, math.e]),
        ([[0, 1], [2, 3]], [[1.0, math.e], [math.e**2, math.e**3]]),
    ],
)
def test_tensor_exp(input, expected):
    x = nt.tensor(input)
    out = x.exp()
    testing.assert_allclose(out, nt.tensor(expected))
    assert out.dtype == nt.tensor(expected).dtype


@pytest.mark.parametrize(
    "input,expected",
    [
        (1.0, 0.0),
        (math.e, 1.0),
        ([1.0, math.e], [0.0, 1.0]),
        ([[1.0, math.e], [math.e**2, math.e**3]], [[0.0, 1.0], [2.0, 3.0]]),
    ],
)
def test_tensor_log(input, expected):
    x = nt.tensor(input)
    out = x.log()
    testing.assert_allclose(out, nt.tensor(expected))
    assert out.dtype == nt.tensor(expected).dtype


@pytest.mark.parametrize(
    "input,exp,expected",
    [
        (1, 1, 1),
        (1, 0, 1),
        (0, 1, 0),
        (0, 0, 1),
        (1.0, 1, 1.0),
        (1, 1.0, 1.0),
        (2, 2, 4),
        (3, 4.0, 81.0),
        (9.0, -2, 0.0123),
        (2.5, 3.7, 29.6741),
        ([1, 2, 3], 2, [1, 4, 9]),
        ([[1, 2], [3, 4]], 2.0, [[1.0, 4.0], [9.0, 16.0]]),
        ([[1.0, 2.0], [3.0, 4.0]], 2, [[1.0, 4.0], [9.0, 16.0]]),
    ],
)
def test_tensor_pow(input, exp, expected):
    x = nt.tensor(input)
    out = x.pow(exp)
    out__pow__ = x**exp
    testing.assert_allclose(out, nt.tensor(expected), tol=1e-4)
    testing.assert_allclose(out, out__pow__)
    assert out.dtype == out__pow__.dtype == nt.tensor(expected).dtype


def test_tensor_pow_int_neg_raises():
    with pytest.raises(RuntimeError):
        _ = nt.tensor(1) ** -2
