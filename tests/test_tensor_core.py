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
