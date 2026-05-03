import math

import pytest
from conftest import requires_cuda

import nanotorch as nt
from nanotorch import testing


@pytest.mark.parametrize(
    "input,expected",
    [
        (0.0, 1.0),
        (1.0, math.e),
        ([0, 1], [1.0, math.e]),
        ([[0, 1], [2, 3]], [[1.0, math.e], [math.e**2, math.e**3]]),
    ],
)
def test_tensor_exp(input, expected):
    x = nt.tensor(input)
    out = x.exp()
    testing.assert_allclose(out, nt.tensor(expected))
    assert out.dtype == nt.tensor(expected).dtype


@pytest.mark.parametrize(
    "input,expected",
    [
        (1.0, 0.0),
        (math.e, 1.0),
        ([1.0, math.e], [0.0, 1.0]),
        ([[1.0, math.e], [math.e**2, math.e**3]], [[0.0, 1.0], [2.0, 3.0]]),
    ],
)
def test_tensor_log(input, expected):
    x = nt.tensor(input)
    out = x.log()
    testing.assert_allclose(out, nt.tensor(expected))
    assert out.dtype == nt.tensor(expected).dtype


@pytest.mark.parametrize(
    "input,exp,expected",
    [
        (1, 1, 1),
        (1, 0, 1),
        (0, 1, 0),
        (0, 0, 1),
        (1.0, 1, 1.0),
        (1, 1.0, 1.0),
        (2, 2, 4),
        (3, 4.0, 81.0),
        (9.0, -2, 0.0123),
        (2.5, 3.7, 29.6741),
        ([1, 2, 3], 2, [1, 4, 9]),
        ([[1, 2], [3, 4]], 2.0, [[1.0, 4.0], [9.0, 16.0]]),
        ([[1.0, 2.0], [3.0, 4.0]], 2, [[1.0, 4.0], [9.0, 16.0]]),
    ],
)
def test_tensor_pow(input, exp, expected):
    x = nt.tensor(input)
    out = x.pow(exp)
    out__pow__ = x**exp
    testing.assert_allclose(out, nt.tensor(expected), tol=1e-4)
    testing.assert_allclose(out, out__pow__)
    assert out.dtype == out__pow__.dtype == nt.tensor(expected).dtype


def test_tensor_pow_int_neg_raises():
    with pytest.raises(RuntimeError):
        _ = nt.tensor(1) ** -2


@pytest.mark.parametrize(
    "input,expected",
    [
        (1.0, 1.0),
        (-1.0, 0.0),
        ([1.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
    ],
)
def test_tensor_relu(input, expected):
    x = nt.tensor(input)
    out = nt.relu(x)
    testing.assert_allclose(out, nt.tensor(expected))
    assert out.dtype == nt.tensor(expected).dtype


# CUDA


@requires_cuda
@pytest.mark.parametrize(
    "op, allow_neg",
    [
        (lambda x: x.exp(), True),
        (lambda x: x.log(), False),
        (lambda x: x.pow(2.0), True),
        (lambda x: x.pow(2.5), False),
        (lambda x: x.pow(-1.3), False),
        (lambda x: -x, True),
        (lambda x: nt.relu(x), True),
    ],
)
def test_cuda_result(op, allow_neg):
    for _ in range(10):
        x = nt.rand(2, 4, 5, dtype=nt.float64)
        if allow_neg:
            x -= 0.5
        assert x.device == nt.Device.Cpu
        result_cpu = op(x)
        x = x.to("cuda")
        result_cuda = op(x)
        assert result_cuda.device == nt.Device.Cuda
        testing.assert_allclose(result_cpu, result_cuda.cpu())


@requires_cuda
@pytest.mark.parametrize(
    "op, allow_neg",
    [
        (lambda x: x.exp(), True),
        (lambda x: x.log(), False),
        (lambda x: x.pow(2.0), True),
        (lambda x: x.pow(2.5), False),
        (lambda x: x.pow(-1.3), False),
        (lambda x: -x, True),
        (lambda x: nt.relu(x), True),
    ],
)
def test_cuda_result_view(op, allow_neg):
    for _ in range(10):
        x = nt.rand(2, 4, 5, dtype=nt.float64)
        if allow_neg:
            x -= 0.5
        x = x[1, :, [1, 2, 4]]
        assert x.device == nt.Device.Cpu
        result_cpu = op(x)
        x = x.to("cuda")
        result_cuda = op(x)
        assert result_cuda.device == nt.Device.Cuda
        testing.assert_allclose(result_cpu, result_cuda.cpu())
