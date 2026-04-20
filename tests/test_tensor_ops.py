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


def test_tensor_sum_empty():
    x = nt.tensor([])
    assert x.sum().tolist() == 0.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_0d():
    x = nt.tensor(2.0)
    assert x.sum().tolist() == 2.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_1d():
    x = nt.tensor([1.0, 2.0, 3.0])
    assert x.sum().tolist() == 6.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_2d():
    x = nt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert x.sum().tolist() == 21.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_3d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert x.sum().tolist() == 36.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_negatives():
    x = nt.tensor([1.0, 2.0, -3.0])
    assert x.sum().tolist() == 0.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_all_negatives():
    x = nt.tensor([-1.0, -2.0, -3.0])
    assert x.sum().tolist() == -6.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_bool():
    x = nt.tensor([True, True, False, True], dtype=nt.DataType.BOOL)
    assert x.sum().tolist() == 3.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_int32():
    x = nt.tensor([1, 7, 0, 8], dtype=nt.DataType.INT32)
    assert x.sum().tolist() == 16.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_int64():
    x = nt.tensor([1, 7, 0, 8], dtype=nt.DataType.INT64)
    assert x.sum().tolist() == 16.0
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_fp32():
    x = nt.tensor([1.5, 7.5, -1.5, 8.0], dtype=nt.DataType.FP32)
    assert x.sum().tolist() == 15.5
    assert x.sum().dtype == nt.DataType.FP32


def test_tensor_sum_fp64():
    x = nt.tensor([1.5, 7.5, -1.5, 8.0], dtype=nt.DataType.FP64)
    assert x.sum().tolist() == 15.5
    assert x.sum().dtype == nt.DataType.FP32


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
    i32 = nt.tensor([1, 2, 3], dtype=nt.DataType.INT32)
    i64 = nt.tensor([1, 2, 3], dtype=nt.DataType.INT64)
    f32 = nt.tensor([1, 2, 3], dtype=nt.DataType.FP32)
    f64 = nt.tensor([1, 2, 3], dtype=nt.DataType.FP64)
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
    assert x.T.to(nt.DataType.FP64).tolist() == [[1, 4], [2, 5], [3, 6]]


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


def test_reshape_empty():
    x = nt.tensor([1, 2, 3])
    with pytest.raises(ValueError):
        x.reshape()


def test_reshape_0d():
    x = nt.tensor(0.0)
    assert x.reshape(1).tolist() == [0.0]


def test_reshape_1d_identity():
    x = nt.tensor([1, 2, 3])
    assert x.reshape(3) is x


def test_reshape_1d_2d():
    x = nt.tensor([1, 2, 3, 4, 5, 6])
    assert x.reshape(2, 3).tolist() == [[1, 2, 3], [4, 5, 6]]


def test_reshape_1d_3d():
    x = nt.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    assert x.reshape(2, 2, 2).tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


def test_reshape_1d_d1():
    x = nt.tensor([1, 2, 3, 4, 5, 6])
    assert x.reshape(1, 6).tolist() == [[1, 2, 3, 4, 5, 6]]


def test_reshape_2d_1d():
    x = nt.tensor([[1, 2, 3], [4, 5, 6]])
    assert x.reshape(6).tolist() == [1, 2, 3, 4, 5, 6]


def test_reshape_3d_1d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert x.reshape(8).tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


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


def test_getitem_chained_3d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert x[1][1][0].tolist() == 7


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
