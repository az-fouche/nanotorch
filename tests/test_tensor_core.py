import pytest

import nanotorch as nt


def test_tensor_init_empty():
    x = nt.tensor([])
    assert x.tolist() == []
    assert x.dtype == nt.float32
    assert x.shape == (0,)


def test_tensor_init_scalar():
    x = nt.tensor(0.0)
    assert x.tolist() == 0.0
    assert x.dtype == nt.float32
    assert x.shape == ()
    assert x.ndim == 0


def test_tensor_init_1d_fp32():
    x = nt.tensor([0.0, 1.0, 2.0, 3.0])
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert x.dtype == nt.float32
    assert x.shape == (4,)


def test_tensor_init_1d_fp64():
    x = nt.tensor([0.0, 1.0, 2.0, 3.0], dtype=nt.float64)
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert x.dtype == nt.float64
    assert x.shape == (4,)


def test_tensor_init_int32():
    x = nt.tensor([0.2, 1.8, 2.1, 3.999], dtype=nt.int32)
    assert x.tolist() == [0, 1, 2, 3]
    assert x.dtype == nt.int32
    assert x.shape == (4,)


def test_tensor_init_int64():
    x = nt.tensor([0.2, 1.8, 2.1, 3.999], dtype=nt.int64)
    assert x.tolist() == [0, 1, 2, 3]
    assert x.dtype == nt.int64
    assert x.shape == (4,)


def test_tensor_init_bool():
    x = nt.tensor([0, 0.001, 2.1, -3.999], dtype=nt.bool_)
    assert x.tolist() == [False, True, True, True]
    assert x.dtype == nt.bool_
    assert x.shape == (4,)


def test_tensor_init_inttofloat():
    x = nt.tensor([0, 1, 3.0, 2])
    assert x.tolist() == [0.0, 1.0, 3.0, 2.0]
    assert x.dtype == nt.float32
    assert x.shape == (4,)


def test_tensor_init_booltofloat():
    x = nt.tensor([True, True, 3.0, True])
    assert x.tolist() == [1.0, 1.0, 3.0, 1.0]
    assert x.dtype == nt.float32
    assert x.shape == (4,)


def test_tensor_init_booltoint():
    x = nt.tensor([True, True, 3, True])
    assert x.tolist() == [1, 1, 3, 1]
    assert x.dtype == nt.int64
    assert x.shape == (4,)
    assert x.ndim == 1


def test_tensor_init_2d():
    x = nt.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert x.tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    assert x.dtype == nt.float32
    assert x.shape == (3, 4)
    assert x.ndim == 2


def test_tensor_init_3d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert x.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert x.dtype == nt.int64
    assert x.shape == (2, 2, 2)
    assert x.ndim == 3


def test_tensor_init_maxrank():
    x32 = nt.tensor([[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]])
    assert x32.ndim == 32


def test_tensor_init_overrank():
    with pytest.raises(ValueError):
        # 33
        nt.tensor([[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]])


def test_tensor_ragged_raises():
    with pytest.raises(ValueError):
        nt.tensor([[1, 2], [3]])
    with pytest.raises(ValueError):
        nt.tensor([[[1, 2]], [[3]]])


def test_tensor_heterogeneous_raises():
    with pytest.raises(ValueError):
        nt.tensor([1, [2]])
    with pytest.raises(ValueError):
        nt.tensor([[1], 2])


def test_tensor_str_raises():
    with pytest.raises(TypeError):
        nt.tensor(["a"])


def test_tensor_to_same():
    x = nt.tensor(1.5, dtype=nt.float32)
    assert x.to(nt.float32) is x


def test_tensor_fp32_to_int32():
    x = nt.tensor(1.5, dtype=nt.float32)
    assert x.to(nt.int32).tolist() == 1


def test_tensor_fp32_to_bool_false():
    x = nt.tensor(0.0, dtype=nt.float32)
    assert x.to(nt.bool_).tolist() is False


def test_tensor_fp32_to_bool_true():
    x = nt.tensor(1.5, dtype=nt.float32)
    assert x.to(nt.bool_).tolist() is True


def test_tensor_bool_to_int_false():
    x = nt.tensor(False, dtype=nt.bool_)
    assert x.to(nt.int32).tolist() == 0


def test_tensor_bool_to_int_true():
    x = nt.tensor(True, dtype=nt.bool_)
    assert x.to(nt.int32).tolist() == 1


def test_tensor_int_to_fp32():
    x = nt.tensor(654891, dtype=nt.int32)
    assert x.to(nt.float32).tolist() == 654891.0


def test_tensor_int_to_int64():
    x = nt.tensor(654891, dtype=nt.int32)
    assert x.to(nt.int64).tolist() == 654891


def test_tensor_int64_to_int():
    x = nt.tensor(654891, dtype=nt.int64)
    assert x.to(nt.int32).tolist() == 654891


def test_tensor_fp32_to_fp64():
    x = nt.tensor(1.5, dtype=nt.float32)
    assert x.to(nt.float64).to(nt.float32).tolist() == 1.5


def test_tensor_fp32_precision_loss():
    x = nt.tensor(0.01, dtype=nt.float64)
    y = x.to(nt.float32).to(nt.float64)
    assert not x.equals(y)


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


def test_getitem_advanced_simple_tensor():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[nt.tensor([1, 2])].tolist() == [[5, 6, 7, 8], [9, 10, 11, 12]]


def test_getitem_advanced_simple_mask():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[False, True, True]].tolist() == [[5, 6, 7, 8], [9, 10, 11, 12]]


def test_getitem_advanced_simple_mask_tensor():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[nt.tensor([False, True, True])].tolist() == [[5, 6, 7, 8], [9, 10, 11, 12]]


def test_getitem_advanced_simple_mask_wrongshape():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    with pytest.raises(IndexError):
        x[[False, True]]


def test_getitem_advanced_neg():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[[-2, -1]].tolist() == [[5, 6, 7, 8], [9, 10, 11, 12]]


def test_getitem_advanced_neg_tensor():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[nt.tensor([-2, -1])].tolist() == [[5, 6, 7, 8], [9, 10, 11, 12]]


def test_getitem_advanced_mixed():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[:, [1, 2]].tolist() == [[2, 3], [6, 7], [10, 11]]


def test_getitem_advanced_mixed_tensor():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[:, nt.tensor([1, 2])].tolist() == [[2, 3], [6, 7], [10, 11]]


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


def test_getitem_advanced_broadcast_tensor():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[nt.tensor([1, 2, 1]), nt.tensor([1, 1, 0])].tolist() == [6, 10, 5]


def test_getitem_advanced_broadcast_error():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    with pytest.raises(ValueError):
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


def test_getitem_advanced_2dindexing_tensor():
    x = nt.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert x[nt.tensor([[0, 1], [1, 2]])].tolist() == [
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[5, 6, 7, 8], [9, 10, 11, 12]],
    ]


def test_getitem_advanced_broadcast_3d():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    assert x[[1, 1], :, [1, 0]].tolist() == [[6, 8], [5, 7]]


def test_getitem_advanced_broadcast_3d_tensor():
    x = nt.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    assert x[nt.tensor([1, 1]), :, nt.tensor([1, 0])].tolist() == [[6, 8], [5, 7]]


def test_getitem_advanced_out_of_bounds():
    x = nt.arange(60).reshape(4, 3, 5)
    with pytest.raises(IndexError):
        x[[1, 4], 2, [2, 3]]  # 4 ≥ axis-0 size
    with pytest.raises(IndexError):
        x[[1, -5]]  # -5 < -4


# __setitem__


def test_setitem_0d_raises():
    x = nt.tensor(0.0)
    with pytest.raises(IndexError):
        x[0] = 1


def test_setitem_1d_single():
    x = nt.arange(6)
    x[2] = 12
    assert x.tolist() == [0, 1, 12, 3, 4, 5]


def test_setitem_1d_single_view():
    x = nt.arange(10)
    y = x[:6]
    y[2] = 12
    assert x.tolist() == [0, 1, 12, 3, 4, 5, 6, 7, 8, 9]
    assert y.tolist() == [0, 1, 12, 3, 4, 5]


def test_setitem_1d_single_neg():
    x = nt.arange(6)
    x[-2] = 12
    assert x.tolist() == [0, 1, 2, 3, 12, 5]


def test_setitem_1d_single_casts():
    x = nt.arange(6)
    x[2] = 12.5
    assert x.tolist() == [0, 1, 12, 3, 4, 5]


def test_setitem_1d_single_fp32():
    x = nt.arange(6, dtype=nt.float32)
    x[2] = 12.5
    assert x.tolist() == [0, 1, 12.5, 3, 4, 5]


def test_setitem_1d_single_bool():
    x = nt.arange(6, dtype=nt.bool_)
    x[2] = False
    assert x.tolist() == [False, True, False, True, True, True]


def test_setitem_1d_single_bool_casts():
    x = nt.arange(6, dtype=nt.bool_)
    x[2] = 0.0
    assert x.tolist() == [False, True, False, True, True, True]


def test_setitem_1d_single_oob_raises():
    x = nt.arange(6)
    with pytest.raises(IndexError):
        x[-7] = 12
    with pytest.raises(IndexError):
        x[6] = 12


def test_setitem_1d_single_shape_raises():
    x = nt.arange(6)
    with pytest.raises(IndexError):
        x[0, 1] = 12


def test_setitem_1d_slice_single():
    x = nt.arange(6)
    x[2:] = 12
    assert x.tolist() == [0, 1, 12, 12, 12, 12]


def test_setitem_1d_slice_single_rev():
    x = nt.arange(6)
    x[:2] = 12
    assert x.tolist() == [12, 12, 2, 3, 4, 5]


def test_setitem_1d_slice_partial():
    x = nt.arange(6)
    x[2:4] = 12
    assert x.tolist() == [0, 1, 12, 12, 4, 5]


def test_setitem_1d_slice_container():
    x = nt.arange(6)
    x[2:] = [12, 13, 1776, 12]
    assert x.tolist() == [0, 1, 12, 13, 1776, 12]


def test_setitem_1d_slice_container_tensor():
    x = nt.arange(6)
    x[2:] = nt.tensor([12, 13, 1776, 12])
    assert x.tolist() == [0, 1, 12, 13, 1776, 12]


def test_setitem_1d_slice_container_small_raises():
    x = nt.arange(6)
    with pytest.raises(IndexError):
        x[2:] = [12, 13, 12]


def test_setitem_1d_slice_container_large_raises():
    x = nt.arange(6)
    with pytest.raises(IndexError):
        x[2:] = [12, 13, 12, 12, 12]


def test_setitem_1d_slice_self():
    x = nt.arange(6)
    x[4:] = x[:2]
    assert x.tolist() == [0, 1, 2, 3, 0, 1]


def test_setitem_1d_boolmask():
    x = nt.arange(6)
    x[[True, False, False, True, False, True]] = 12
    assert x.tolist() == [12, 1, 2, 12, 4, 12]


def test_setitem_1d_boolmask_wrongshape():
    x = nt.arange(6)
    with pytest.raises(IndexError):
        x[[True, False, False, True, False]] = 12


def test_setitem_1d_multiindex():
    x = nt.arange(6)
    x[[1, 3, 5]] = 12
    assert x.tolist() == [0, 12, 2, 12, 4, 12]


def test_setitem_1d_multiindex_tensor():
    x = nt.arange(6)
    x[nt.tensor([1, 3, 5])] = 12
    assert x.tolist() == [0, 12, 2, 12, 4, 12]


def test_setitem_1d_multiindex_neg():
    x = nt.arange(6)
    x[[1, 3, -1]] = 12
    assert x.tolist() == [0, 12, 2, 12, 4, 12]


def test_setitem_1d_multiindex_repeat_lastwins():
    x = nt.arange(6)
    x[[0, 0, 0]] = [12, 13, 14]
    assert x.tolist() == [14, 1, 2, 3, 4, 5]


def test_setitem_1d_multiindex_oob():
    x = nt.arange(6)
    with pytest.raises(IndexError):
        x[[1, 3, 5, 12]] = 12


def test_setitem_2d_pointwise():
    x = nt.arange(12).reshape(3, 4)
    x[0, 2] = 12
    assert x.tolist() == [[0, 1, 12, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_setitem_2d_single():
    x = nt.arange(12).reshape(3, 4)
    x[2] = 12
    assert x.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [12, 12, 12, 12]]


def test_setitem_2d_single_col():
    x = nt.arange(12).reshape(3, 4)
    x[:, 2] = 12
    assert x.tolist() == [[0, 1, 12, 3], [4, 5, 12, 7], [8, 9, 12, 11]]


def test_setitem_2d_single_view():
    x = nt.arange(24).reshape(6, 4)
    y = x[:3, :3]
    y[:, 2] = 12
    assert x.tolist() == [
        [0, 1, 12, 3],
        [4, 5, 12, 7],
        [8, 9, 12, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
    ]
    assert y.tolist() == [[0, 1, 12], [4, 5, 12], [8, 9, 12]]


def test_setitem_2d_single_neg():
    x = nt.arange(12).reshape(3, 4)
    x[-2] = 12
    assert x.tolist() == [[0, 1, 2, 3], [12, 12, 12, 12], [8, 9, 10, 11]]


def test_setitem_2d_single_oob_raises():
    x = nt.arange(12).reshape(3, 4)
    with pytest.raises(IndexError):
        x[-4] = 12
    with pytest.raises(IndexError):
        x[3] = 12


def test_setitem_2d_single_container():
    x = nt.arange(12).reshape(3, 4)
    x[2] = [123, 54, 8, 4]
    assert x.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [123, 54, 8, 4]]


def test_setitem_2d_single_container_small_raises():
    x = nt.arange(12).reshape(3, 4)
    with pytest.raises(IndexError):
        x[2] = [123, 54, 8]


def test_setitem_2d_single_container_large_raises():
    x = nt.arange(12).reshape(3, 4)
    with pytest.raises(IndexError):
        x[2] = [123, 54, 8, 12, 12]


def test_setitem_2d_single_container_shape_raises():
    x = nt.arange(12).reshape(3, 4)
    with pytest.raises(IndexError):
        x[2] = [[123, 54, 8], [12, 12, 12]]


def test_setitem_2d_slice_single():
    x = nt.arange(12).reshape(3, 4)
    x[1:] = 12
    assert x.tolist() == [[0, 1, 2, 3], [12, 12, 12, 12], [12, 12, 12, 12]]


def test_setitem_2d_slice_single_rev():
    x = nt.arange(12).reshape(3, 4)
    x[:2] = 12
    assert x.tolist() == [[12, 12, 12, 12], [12, 12, 12, 12], [8, 9, 10, 11]]


def test_setitem_2d_slice_partial():
    x = nt.arange(12).reshape(4, 3)
    x[1:3] = 12
    assert x.tolist() == [[0, 1, 2], [12, 12, 12], [12, 12, 12], [9, 10, 11]]


def test_setitem_2d_slice_container():
    x = nt.arange(12).reshape(4, 3)
    x[1:3] = [65, 5, 9]
    assert x.tolist() == [[0, 1, 2], [65, 5, 9], [65, 5, 9], [9, 10, 11]]


def test_setitem_2d_ellipsis():
    x = nt.arange(12).reshape(4, 3)
    x[..., 2] = 12
    assert x.tolist() == [[0, 1, 12], [3, 4, 12], [6, 7, 12], [9, 10, 12]]


def test_setitem_2d_boolmask():
    x = nt.arange(12).reshape(4, 3)
    x[[True, False, False, True]] = 12
    assert x.tolist() == [[12, 12, 12], [3, 4, 5], [6, 7, 8], [12, 12, 12]]


def test_setitem_2d_boolmask_wrongshape():
    x = nt.arange(12).reshape(4, 3)
    with pytest.raises(IndexError):
        x[[True, False, False]] = 12


def test_setitem_2d_multiindex():
    x = nt.arange(12).reshape(4, 3)
    x[[0, 3]] = 12
    assert x.tolist() == [[12, 12, 12], [3, 4, 5], [6, 7, 8], [12, 12, 12]]


def test_setitem_2d_multiindex_neg():
    x = nt.arange(12).reshape(4, 3)
    x[[0, -1]] = 12
    assert x.tolist() == [[12, 12, 12], [3, 4, 5], [6, 7, 8], [12, 12, 12]]


def test_setitem_2d_multiindex_oob():
    x = nt.arange(12).reshape(4, 3)
    with pytest.raises(IndexError):
        x[[0, 4]] = 12


def test_setitem_2d_mixed_index():
    x = nt.arange(12).reshape(3, 4)
    x[:, [1, 2]] = [[10, 20], [30, 40], [50, 60]]
    assert x.tolist() == [[0, 10, 20, 3], [4, 30, 40, 7], [8, 50, 60, 11]]


def test_setitem_2d_advanced_tensor():
    x = nt.arange(12).reshape(3, 4)
    x[[[0, 1], [1, 2]]] = 33
    assert x.tolist() == [[33, 33, 33, 33], [33, 33, 33, 33], [33, 33, 33, 33]]


def test_setitem_2d_advanced_broadcast():
    x = nt.arange(12).reshape(3, 4)
    x[[0, 1], [1, 2]] = 33
    assert x.tolist() == [[0, 33, 2, 3], [4, 5, 33, 7], [8, 9, 10, 11]]
