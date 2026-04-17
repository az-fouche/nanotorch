import nanotorch as nt


def test_tensor_init_empty():
    x = nt.Tensor([])
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (0,)


def test_tensor_init_scalar():
    x = nt.Tensor(0.0)
    assert x.dtype == nt.DataType.FP32
    assert x.shape == ()


def test_tensor_init_1d_fp32():
    x = nt.Tensor([0.0, 1.0, 2.0, 3.0])
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (4,)


def test_tensor_init_1d_fp64():
    x = nt.Tensor([0.0, 1.0, 2.0, 3.0], dtype=nt.DataType.FP64)
    assert x.dtype == nt.DataType.FP64
    assert x.shape == (4,)


def test_tensor_init_int32():
    x = nt.Tensor([0.0, 1.0, 2.0, 3.0], dtype=nt.DataType.INT32)
    assert x.dtype == nt.DataType.INT32
    assert x.shape == (4,)


def test_tensor_init_int64():
    x = nt.Tensor([0.0, 1.0, 2.0, 3.0], dtype=nt.DataType.INT64)
    assert x.dtype == nt.DataType.INT64
    assert x.shape == (4,)


def test_tensor_init_bool():
    x = nt.Tensor([0.0, 1.0, 2.0, 3.0], dtype=nt.DataType.BOOL)
    assert x.dtype == nt.DataType.BOOL
    assert x.shape == (4,)


def test_tensor_init_inttofloat():
    x = nt.Tensor([0, 1, 3.0, 2])
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (4,)


def test_tensor_init_booltofloat():
    x = nt.Tensor([True, True, 3.0, True])
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (4,)


def test_tensor_init_booltoint():
    x = nt.Tensor([True, True, 3, True])
    assert x.dtype == nt.DataType.INT64
    assert x.shape == (4,)


def test_tensor_init_2d():
    x = nt.Tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert x.dtype == nt.DataType.FP32
    assert x.shape == (3, 4)


def test_tensor_init_cast():
    x = nt.Tensor([0.0, 1.0, 2.0, 3.0], dtype=nt.DataType.INT32)
    assert x.dtype == nt.DataType.INT32
    assert x.shape == (4,)


def test_tensor_sum_0d():
    x = nt.Tensor(2.0)
    assert x.sum() == 2.0


def test_tensor_sum_1d():
    x = nt.Tensor([1.0, 2.0, 3.0])
    assert x.sum() == 6.0


def test_tensor_sum_2d():
    x = nt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert x.sum() == 21.0
