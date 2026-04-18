import nanotorch as nt


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


def test_tensor_equal_shape():
    x = nt.tensor([1, 2, 3])
    y = nt.tensor([1, 2])
    assert not x.equals(y)


def test_tensor_equal_type():
    i32 = nt.tensor([1, 2, 3], dtype=nt.DataType.INT32)
    i64 = nt.tensor([1, 2, 3], dtype=nt.DataType.INT64)
    f32 = nt.tensor([1, 2, 3], dtype=nt.DataType.FP32)
    f64 = nt.tensor([1, 2, 3], dtype=nt.DataType.FP64)
    assert i32.equals(i64)
    assert i32.equals(f32)
    assert i32.equals(f64)


def test_tensor_equal_empty():
    assert nt.tensor([]).equals(nt.tensor([]))


def test_tensor_unequal():
    x = nt.tensor([1, 2, 3, 4, 5])
    y = nt.tensor([1, 2, 2, 4, 5])
    assert not x.equals(y)


def test_tensor_unequal_close():
    x = nt.tensor([1, 2, 3, 4, 5])
    y = nt.tensor([1, 2, 3.00001, 4, 5])
    assert not x.equals(y)
