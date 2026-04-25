import numpy as np
import pytest

import nanotorch as nt

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
        (([True, True, False, True],), 3.0),
        (([1, 7, 0, 8], nt.int32), 16.0),
        (([1, 7, 0, 8], nt.int64), 16.0),
        (([1.5, 7.5, -1.5, 8.0], nt.float32), 15.5),
        (([1.5, 7.5, -1.5, 8.0], nt.float64), 15.5),
    ],
)
def test_tensor_sum(input, expected):
    x = nt.tensor(*input)
    assert x.sum().tolist() == expected
    assert x.sum().dtype == nt.float32


def test_tensor_sum_transpose():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert x.sum().tolist() == x.T.sum().tolist()


def test_tensor_sum_1d_intindex_slice():
    x = nt.tensor([1, 2, 3, 4, 5, 6])
    assert x[3].sum().tolist() == 4


def test_tensor_sum_2d_intindex_slice():
    x = nt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert x[1].sum().tolist() == 15


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
    assert x.T is x


def test_tensor_transpose_0d():
    x = nt.tensor(0.0)
    assert x.T is x


def test_tensor_transpose_1d():
    x = nt.tensor([0.0, 1.0, 2.0, 3.0])
    assert x.T is x


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


# __getitem__: single index


def test_getitem_intindex_0d():
    x = nt.tensor(0.0)
    with pytest.raises(IndexError):
        x[0]


def test_getitem_intindex_toomany_d():
    x = nt.tensor([0.0, 1.0, 2.0])
    with pytest.raises(IndexError):
        x[0, 0]


def test_getitem_intindex_1d():
    x = nt.tensor([6, 5, 4, 3, 2, 1])
    for i in range(6):
        assert x[i].tolist() == 6 - i


def test_getitem_intindex_1d_wrap():
    x = nt.tensor([6, 5, 4, 3, 2, 1])
    for i in range(6):
        assert x[-i - 1].tolist() == i + 1


def test_getitem_intindex_1d_oob():
    x = nt.tensor([6, 5, 4, 3, 2, 1])
    with pytest.raises(IndexError):
        x[6]


def test_getitem_intindex_1d_wrap_oob():
    x = nt.tensor([6, 5, 4, 3, 2, 1])
    with pytest.raises(IndexError):
        x[-7]


def test_getitem_slice_1d():
    x = nt.tensor([1, 3, 4, 5, 6, 9])
    assert x[2:5].tolist() == [4, 5, 6]


def test_getitem_slice_1d_aliasing():
    x = nt.tensor([1, 3, 4, 5, 6, 9])
    y = x[1:3]
    assert y.tolist() == [3, 4]
    data = memoryview(x._data)
    data[1] = 12
    assert y.tolist() == [12, 4]


def test_getitem_slice_1d_min():
    x = nt.tensor([1, 3, 4, 5, 6, 9])
    assert x[:3].tolist() == [1, 3, 4]


def test_getitem_slice_1d_max():
    x = nt.tensor([1, 3, 4, 5, 6, 9])
    assert x[3:].tolist() == [5, 6, 9]


def test_getitem_slice_1d_rev():
    x = nt.tensor([1, 3, 4, 5, 6, 9])
    assert x[::-1].tolist() == [9, 6, 5, 4, 3, 1]


def test_getitem_newaxis_1d_0():
    x = nt.tensor([1, 3, 4, 5, 6, 9])
    assert x[None, :].tolist() == [[1, 3, 4, 5, 6, 9]]


def test_getitem_newaxis_1d_1():
    x = nt.tensor([1, 3, 4, 5, 6, 9])
    assert x[:, None].tolist() == [[1], [3], [4], [5], [6], [9]]


def test_getitem_newaxis_1d_01():
    x = nt.tensor([1, 3, 4, 5, 6, 9])
    assert x[None, :, None].tolist() == [[[1], [3], [4], [5], [6], [9]]]


def test_getitem_intindex_2d():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[0].tolist() == [1, 2, 3, 4]
    assert x[1].tolist() == [5, 6, 7, 8]
    assert x[2].tolist() == [9, 10, 11, 12]
    assert x[0, 0].tolist() == 1
    assert x[0, 0].ndim == 0


def test_getitem_intindex_2d_oob():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    with pytest.raises(IndexError):
        x[3]


def test_getitem_intindex_2d_wrap():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[-3].tolist() == [1, 2, 3, 4]
    assert x[-2].tolist() == [5, 6, 7, 8]
    assert x[-1].tolist() == [9, 10, 11, 12]


def test_getitem_intindex_2d_wrap_oob():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    with pytest.raises(IndexError):
        x[-4]


def test_getitem_slice_2d():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[0:2].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert x[1:, 2:].tolist() == [[7, 8], [11, 12]]
    assert x[2, 1].tolist() == 10
    assert x[1, 2:].tolist() == [7, 8]
    assert x[1, ::-1].tolist() == [8, 7, 6, 5]


def test_getitem_intindex_2d_transpose():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).T
    assert x[0].tolist() == [1, 5, 9]
    assert x[1].tolist() == [2, 6, 10]
    assert x[2].tolist() == [3, 7, 11]
    assert x[3].tolist() == [4, 8, 12]


def test_getitem_intindex_2d_reshape():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).reshape(6, 2)
    assert x[0].tolist() == [1, 2]
    assert x[1].tolist() == [3, 4]
    assert x[2].tolist() == [5, 6]
    assert x[3].tolist() == [7, 8]
    assert x[4].tolist() == [9, 10]
    assert x[5].tolist() == [11, 12]


def test_getitem_3d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert x[0].tolist() == [[1, 2], [3, 4]]
    assert x[0, 1, 1].tolist() == 4
    assert x[0, :, 1].tolist() == [2, 4]
    assert x[:, 1, :].tolist() == [[3, 4], [7, 8]]
    assert x[:, None, :, None].tolist() == [
        [[[[1, 2]], [[3, 4]]]],
        [[[[5, 6]], [[7, 8]]]],
    ]


def test_getitem_chained_3d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert x[1][1][0].tolist() == 7


def test_getitem_slice_advanced():
    x = nt.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    assert x[::2].tolist() == [0, 2, 4, 6]
    assert x[1::2].tolist() == [1, 3, 5, 7]
    assert x[1:5:2].tolist() == [1, 3]
    assert x[2:2].tolist() == []
    assert x[10:5].tolist() == []
    assert x[-3:-1].tolist() == [5, 6]
    assert x[-3::-1].tolist() == [5, 4, 3, 2, 1, 0]
    assert x[0:100].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert x[1:4][1:2].tolist() == [2]


def test_getitem_empty():
    x = nt.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    assert x[()] is not x
    assert x[()].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_getitem_ellipsis():
    x = nt.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
    assert x[...].tolist() == x.tolist()
    assert x[..., 0].tolist() == x[:, :, :, 0].tolist()
    assert x[..., 0, None].tolist() == x[:, :, :, 0, None].tolist()
    assert x[0, ...].tolist() == x[0, :, :, :].tolist()
    assert x[0, ..., 0].tolist() == x[0, :, :, 0].tolist()
    assert x[0, ..., None, 0].tolist() == x[0, :, :, None, 0].tolist()


def test_getitem_ellipsis_multi_raises():
    x = nt.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
    with pytest.raises(IndexError):
        x[..., 0, ...]


def test_getitem_advanced_simple():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[1, 2]].tolist() == [[5, 6, 7, 8], [9, 10, 11, 12]]


def test_getitem_advanced_neg():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[-2, -1]].tolist() == [[5, 6, 7, 8], [9, 10, 11, 12]]


def test_getitem_advanced_mixed():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[:, [1, 2]].tolist() == [[2, 3], [6, 7], [10, 11]]


def test_getitem_advanced_oob():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    with pytest.raises(IndexError):
        x[3]


def test_getitem_advanced_empty():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[]].tolist() == []
    assert x[[]].shape == (0, 4)


def test_getitem_advanced_broadcast():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[1, 2, 1], [1, 1, 0]].tolist() == [6, 10, 5]


def test_getitem_advanced_broadcast_error():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    with pytest.raises(IndexError):
        x[[1, 2, 1], [1, 1, 0, 1]]


def test_getitem_advanced_broadcast_implicit_l():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[1, 2, 1], [1]].tolist() == [6, 10, 6]


def test_getitem_advanced_broadcast_implicit_r():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[1], [1, 2, 1]].tolist() == [6, 7, 6]


def test_getitem_advanced_2dindexing():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[[0, 1], [1, 2]]].tolist() == [
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[5, 6, 7, 8], [9, 10, 11, 12]],
    ]


def test_getitem_advanced_broadcast_3d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    assert x[[1, 1], :, [1, 0]].tolist() == [[6, 8], [5, 7]]


def test_getitem_advanced_out_of_bounds():
    x = nt.arange(60).reshape(4, 3, 5)
    with pytest.raises(IndexError):
        x[[1, 4], 2, [2, 3]]  # 4 ≥ axis-0 size
    with pytest.raises(IndexError):
        x[[1, -5]]  # -5 < -4


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
