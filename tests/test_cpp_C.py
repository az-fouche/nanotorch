from nanotorch import _C


def test_cpp_sum_b():
    x = _C.Storage.from_iterable([True, False, True], _C.Dtype.Bool)
    assert _C.sum(x) == 2.0


def test_cpp_sum_i32():
    x = _C.Storage.from_iterable([1, 2, 3], _C.Dtype.Int32)
    assert _C.sum(x) == 6.0


def test_cpp_sum_i64():
    x = _C.Storage.from_iterable([1, 2, 3], _C.Dtype.Int64)
    assert _C.sum(x) == 6.0


def test_cpp_sum_f32():
    x = _C.Storage.from_iterable([1.0, 2.0, 3.0], _C.Dtype.Float32)
    assert _C.sum(x) == 6.0


def test_cpp_sum_f64():
    x = _C.Storage.from_iterable([1.0, 2.0, 3.0], _C.Dtype.Float64)
    assert _C.sum(x) == 6.0
