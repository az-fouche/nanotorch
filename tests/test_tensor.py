import pytest

import nanotorch as nt


def test_tensor_init_empty():
    x = nt.tensor([])
    assert x.tolist() == []
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (0,)


def test_tensor_init_scalar():
    x = nt.tensor(0.0)
    assert x.tolist() == 0.0
    assert x.dtype == nt.DataType.FP32
    assert x.shape == ()


def test_tensor_init_1d_fp32():
    x = nt.tensor([0.0, 1.0, 2.0, 3.0])
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (4,)


def test_tensor_init_1d_fp64():
    x = nt.tensor([0.0, 1.0, 2.0, 3.0], dtype=nt.DataType.FP64)
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert x.dtype == nt.DataType.FP64
    assert x.shape == (4,)


def test_tensor_init_int32():
    x = nt.tensor([0.2, 1.8, 2.1, 3.999], dtype=nt.DataType.INT32)
    assert x.tolist() == [0, 1, 2, 3]
    assert x.dtype == nt.DataType.INT32
    assert x.shape == (4,)


def test_tensor_init_int64():
    x = nt.tensor([0.2, 1.8, 2.1, 3.999], dtype=nt.DataType.INT64)
    assert x.tolist() == [0, 1, 2, 3]
    assert x.dtype == nt.DataType.INT64
    assert x.shape == (4,)


def test_tensor_init_bool():
    x = nt.tensor([0, 0.001, 2.1, -3.999], dtype=nt.DataType.BOOL)
    assert x.tolist() == [False, True, True, True]
    assert x.dtype == nt.DataType.BOOL
    assert x.shape == (4,)


def test_tensor_init_inttofloat():
    x = nt.tensor([0, 1, 3.0, 2])
    assert x.tolist() == [0.0, 1.0, 3.0, 2.0]
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (4,)


def test_tensor_init_booltofloat():
    x = nt.tensor([True, True, 3.0, True])
    assert x.tolist() == [1.0, 1.0, 3.0, 1.0]
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (4,)


def test_tensor_init_booltoint():
    x = nt.tensor([True, True, 3, True])
    assert x.tolist() == [1, 1, 3, 1]
    assert x.dtype == nt.DataType.INT64
    assert x.shape == (4,)


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
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (3, 4)


def test_tensor_ragged_raises():
    with pytest.raises(ValueError):
        nt.tensor([[1, 2], [3]])


def test_tensor_3d_raises():
    with pytest.raises(ValueError):
        nt.tensor([[[1]]])


def test_tensor_str_raises():
    with pytest.raises(TypeError):
        nt.tensor(["a"])


def test_tensor_to_same():
    x = nt.tensor(1.5, dtype=nt.DataType.FP32)
    assert x.to(nt.DataType.FP32) is x


def test_tensor_fp32_to_int32():
    x = nt.tensor(1.5, dtype=nt.DataType.FP32)
    assert x.to(nt.DataType.INT32).tolist() == 1


def test_tensor_fp32_to_bool_false():
    x = nt.tensor(0.0, dtype=nt.DataType.FP32)
    assert x.to(nt.DataType.BOOL).tolist() is False


def test_tensor_fp32_to_bool_true():
    x = nt.tensor(1.5, dtype=nt.DataType.FP32)
    assert x.to(nt.DataType.BOOL).tolist() is True


def test_tensor_bool_to_int_false():
    x = nt.tensor(False, dtype=nt.DataType.BOOL)
    assert x.to(nt.DataType.INT32).tolist() == 0


def test_tensor_bool_to_int_true():
    x = nt.tensor(True, dtype=nt.DataType.BOOL)
    assert x.to(nt.DataType.INT32).tolist() == 1


def test_tensor_int_to_fp32():
    x = nt.tensor(654891, dtype=nt.DataType.INT32)
    assert x.to(nt.DataType.FP32).tolist() == 654891.0


def test_tensor_int_to_int64():
    x = nt.tensor(654891, dtype=nt.DataType.INT32)
    assert x.to(nt.DataType.INT64).tolist() == 654891


def test_tensor_int64_to_int():
    x = nt.tensor(654891, dtype=nt.DataType.INT64)
    assert x.to(nt.DataType.INT32).tolist() == 654891


def test_tensor_fp32_to_fp64():
    x = nt.tensor(1.5, dtype=nt.DataType.FP32)
    assert x.to(nt.DataType.FP64).to(nt.DataType.FP32).tolist() == 1.5


def test_tensor_fp32_precision_loss():
    x = nt.tensor(0.01, dtype=nt.DataType.FP64)
    y = x.to(nt.DataType.FP32).to(nt.DataType.FP64)
    assert not x.equals(y)
