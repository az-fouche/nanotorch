import pytest

import nanotorch as nt


@pytest.mark.parametrize("factory", [nt.zeros, nt.ones])
def test_empty(factory):
    x = factory(0)
    assert x.tolist() == []
    assert x.shape == (0,)
    assert x.dtype == nt.float32


@pytest.mark.parametrize(
    "factory,expected",
    [
        (nt.zeros, 0.0),
        (nt.ones, 1.0),
        (lambda shape: nt.full(shape, 3.14), 3.14),
    ],
)
def test_zeros_scalar(factory, expected):
    x = factory(())
    assert x.tolist() == pytest.approx(expected)
    assert x.shape == ()
    assert x.dtype == nt.float32


@pytest.mark.parametrize(
    "factory,expected",
    [
        (nt.zeros, 0.0),
        (nt.ones, 1.0),
        (lambda shape: nt.full(shape, 3.14), 3.14),
    ],
)
def test_zeros_1d(factory, expected):
    x = factory((3,))
    assert x.tolist() == [pytest.approx(expected) for _ in range(3)]
    assert x.shape == (3,)
    assert x.dtype == nt.float32


@pytest.mark.parametrize(
    "factory,expected",
    [
        (nt.zeros, 0.0),
        (nt.ones, 1.0),
        (lambda shape: nt.full(shape, 3.14), 3.14),
    ],
)
def test_zeros_2d(factory, expected):
    x = factory((3, 5))
    assert x.tolist() == [[pytest.approx(expected) for _ in range(5)] for _ in range(3)]
    assert x.shape == (3, 5)
    assert x.dtype == nt.float32


@pytest.mark.parametrize(
    "factory,expected",
    [
        (nt.zeros, 0.0),
        (nt.ones, 1.0),
        (lambda shape: nt.full(shape, 3.14), 3.14),
    ],
)
def test_zeros_3d(factory, expected):
    x = factory((3, 5, 2))
    assert x.tolist() == [
        [[pytest.approx(expected) for _ in range(2)] for _ in range(5)]
        for _ in range(3)
    ]
    assert x.shape == (3, 5, 2)
    assert x.dtype == nt.float32


@pytest.mark.parametrize(
    "factory,expected,dtype",
    [
        (nt.zeros, False, nt.bool_),
        (nt.zeros, 0, nt.int32),
        (nt.zeros, 0, nt.int64),
        (nt.zeros, 0.0, nt.float32),
        (nt.zeros, 0.0, nt.float64),
        (nt.ones, True, nt.bool_),
        (nt.ones, 1, nt.int32),
        (nt.ones, 1, nt.int64),
        (nt.ones, 1.0, nt.float32),
        (nt.ones, 1.0, nt.float64),
        (lambda shape, dtype: nt.full(shape, 3.14, dtype), True, nt.bool_),
        (lambda shape, dtype: nt.full(shape, 3.14, dtype), 3, nt.int32),
        (lambda shape, dtype: nt.full(shape, 3.14, dtype), 3, nt.int64),
        (lambda shape, dtype: nt.full(shape, 3.14, dtype), 3.14, nt.float32),
        (lambda shape, dtype: nt.full(shape, 3.14, dtype), 3.14, nt.float64),
    ],
)
def test_zeros_dtype(factory, expected, dtype):
    x = factory((3,), dtype=dtype)
    assert x.tolist() == [pytest.approx(expected) for _ in range(3)]
    assert x.shape == (3,)
    assert x.dtype == dtype


def test_eye_empty():
    x = nt.eye(0)
    assert x.tolist() == []
    assert x.shape == (0, 0)
    assert x.dtype == nt.float32


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (nt.bool_, True),
        (nt.int32, 1),
        (nt.int64, 1),
        (nt.float32, 1.0),
        (nt.float64, 1.0),
    ],
)
def test_eye_1(dtype, expected):
    x = nt.eye(1, dtype=dtype)
    assert x.tolist() == [[expected]]
    assert x.shape == (1, 1)
    assert x.dtype == dtype


def test_eye_3():
    x = nt.eye(3)
    assert x.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert x.shape == (3, 3)
    assert x.dtype == nt.float32


def test_arange_empty():
    x = nt.arange(0)
    assert x.tolist() == []
    assert x.shape == (0,)
    assert x.dtype == nt.int64


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (nt.bool_, [False, True, True, True, True]),
        (nt.int32, [0, 1, 2, 3, 4]),
        (nt.int64, [0, 1, 2, 3, 4]),
        (nt.float32, [0.0, 1.0, 2.0, 3.0, 4.0]),
        (nt.float64, [0.0, 1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_arange_5(dtype, expected):
    x = nt.arange(5, dtype=dtype)
    assert x.tolist() == expected
    assert x.shape == (5,)
    assert x.dtype == dtype


def test_arange_start_step():
    x = nt.arange(5, start=7, step=3)
    assert x.tolist() == [7, 10, 13, 16, 19]
    assert x.shape == (5,)
    assert x.dtype == nt.int64
