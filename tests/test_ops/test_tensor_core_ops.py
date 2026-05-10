import numpy as np
import pytest
from conftest import requires_cuda

import nanotorch as nt

# Tolist


def test_tolist():
    x = nt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert x.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_tolist_extensive():
    # Using numpy here allows to check against a "ground truth"
    np.random.seed(42)
    for _ in range(100):
        ndim = np.random.randint(0, 5)
        shape = np.random.randint(0, 5, (ndim,))
        xt = np.random.randint(-10, 10, shape).tolist()
        assert nt.tensor(xt).tolist() == xt


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


@requires_cuda
def test_tensor_unequal_cuda():
    x = nt.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device="cuda")
    y = nt.tensor([1, 2, 2, 4, 5, 6, 7, 8, 9, 10], device="cuda")
    assert not x.equals(y)


@requires_cuda
def test_tensor_equal_cuda():
    x = nt.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device="cuda")
    y = nt.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device="cuda")
    assert x.equals(y)


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
    assert x.reshape(3).storage is x.storage


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
