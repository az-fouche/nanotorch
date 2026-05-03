import pytest
import torch

import nanotorch as nt
from nanotorch import testing

OPS_SPEC = [
    # add
    (nt.tensor(0.0), 3.14, nt.add, nt.tensor(3.14)),
    (3.14, nt.tensor(0.0), nt.add, nt.tensor(3.14)),
    (nt.tensor(0.0), -3.14, nt.add, nt.tensor(-3.14)),
    (nt.tensor(0.0), True, nt.add, nt.tensor(1.0)),
    (nt.tensor(False), 3.14, nt.add, nt.tensor(3.14)),
    (nt.zeros(5), 3.14, nt.add, nt.full(5, value=3.14)),
    (nt.zeros(5, 7), 3.14, nt.add, nt.full(5, 7, value=3.14)),
    (nt.zeros(5, 7), nt.full(5, 7, value=3.14), nt.add, nt.full(5, 7, value=3.14)),
    (
        nt.zeros(5, 7, dtype=nt.bool_),
        nt.full(5, 7, value=3.14),
        nt.add,
        nt.full(5, 7, value=3.14),
    ),
    (nt.zeros(5, 7), nt.full(5, 1, value=3.14), nt.add, nt.full(5, 7, value=3.14)),
    (nt.zeros(5, 7), nt.full(7, value=3.14), nt.add, nt.full(5, 7, value=3.14)),
    (nt.zeros(7, 5).T, nt.full(7, value=3.14), nt.add, nt.full(5, 7, value=3.14)),
    # subtract
    (nt.tensor(0.0), 3.14, nt.subtract, nt.tensor(-3.14)),
    (3.14, nt.tensor(0.0), nt.subtract, nt.tensor(3.14)),
    (nt.tensor(0.0), -3.14, nt.subtract, nt.tensor(3.14)),
    (nt.tensor(0.0), True, nt.subtract, nt.tensor(-1.0)),
    (nt.tensor(False), 3.14, nt.subtract, nt.tensor(-3.14)),
    (nt.zeros(5), 3.14, nt.subtract, nt.full(5, value=-3.14)),
    (nt.zeros(5, 7), 3.14, nt.subtract, nt.full(5, 7, value=-3.14)),
    (
        nt.zeros(5, 7),
        nt.full(5, 7, value=3.14),
        nt.subtract,
        nt.full(5, 7, value=-3.14),
    ),
    (
        nt.zeros(5, 7, dtype=nt.bool_),
        nt.full(5, 7, value=3.14),
        nt.subtract,
        nt.full(5, 7, value=-3.14),
    ),
    (
        nt.zeros(5, 7),
        nt.full(5, 1, value=3.14),
        nt.subtract,
        nt.full(5, 7, value=-3.14),
    ),
    (
        nt.zeros(5, 7),
        nt.full(7, value=3.14),
        nt.subtract,
        nt.full(5, 7, value=-3.14),
    ),
    (
        nt.zeros(7, 5).T,
        nt.full(7, value=3.14),
        nt.subtract,
        nt.full(5, 7, value=-3.14),
    ),
    # multiply
    (nt.tensor(1.0), 3.14, nt.multiply, nt.tensor(3.14)),
    (3.14, nt.tensor(1.0), nt.multiply, nt.tensor(3.14)),
    (nt.tensor(1.0), -3.14, nt.multiply, nt.tensor(-3.14)),
    (nt.tensor(1.0), True, nt.multiply, nt.tensor(1.0)),
    (nt.tensor(True), 3.14, nt.multiply, nt.tensor(3.14)),
    (nt.ones(5), 3.14, nt.multiply, nt.full(5, value=3.14)),
    (nt.ones(5, 7), 3.14, nt.multiply, nt.full(5, 7, value=3.14)),
    (
        nt.ones(5, 7),
        nt.full(5, 7, value=3.14),
        nt.multiply,
        nt.full(5, 7, value=3.14),
    ),
    (
        nt.ones(5, 7, dtype=nt.bool_),
        nt.full(5, 7, value=3.14),
        nt.multiply,
        nt.full(5, 7, value=3.14),
    ),
    (
        nt.ones(5, 7),
        nt.full(5, 1, value=3.14),
        nt.multiply,
        nt.full(5, 7, value=3.14),
    ),
    (nt.ones(5, 7), nt.full(7, value=3.14), nt.multiply, nt.full(5, 7, value=3.14)),
    (
        nt.ones(7, 5).T,
        nt.full(7, value=3.14),
        nt.multiply,
        nt.full(5, 7, value=3.14),
    ),
    # divide
    (nt.tensor(6.28), 2.0, nt.divide, nt.tensor(3.14)),
    (6.28, nt.tensor(2.0), nt.divide, nt.tensor(3.14)),
    (nt.tensor(6.28), -2.0, nt.divide, nt.tensor(-3.14)),
    (nt.tensor(3.14), True, nt.divide, nt.tensor(3.14)),
    (nt.tensor(False), 2.0, nt.divide, nt.tensor(0.0)),
    (nt.full(5, value=6.28), 2.0, nt.divide, nt.full(5, value=3.14)),
    (nt.full(5, 7, value=6.28), 2.0, nt.divide, nt.full(5, 7, value=3.14)),
    (
        nt.full(5, 7, value=6.28),
        nt.full(5, 7, value=2.0),
        nt.divide,
        nt.full(5, 7, value=3.14),
    ),
    (
        nt.zeros(5, 7, dtype=nt.bool_),
        nt.full(5, 7, value=2.0),
        nt.divide,
        nt.zeros(5, 7),
    ),
    (
        nt.full(5, 7, value=6.28),
        nt.full(5, 1, value=2.0),
        nt.divide,
        nt.full(5, 7, value=3.14),
    ),
    (
        nt.full(5, 7, value=6.28),
        nt.full(7, value=2.0),
        nt.divide,
        nt.full(5, 7, value=3.14),
    ),
    (
        nt.full(7, 5, value=6.28).T,
        nt.full(7, value=2.0),
        nt.divide,
        nt.full(5, 7, value=3.14),
    ),
    # matmul
    (
        nt.tensor([[1.0, 2.0], [3.0, 4.0]]),
        nt.tensor([[5.0, 6.0], [7.0, 8.0]]),
        nt.matmul,
        nt.tensor([[19.0, 22.0], [43.0, 50.0]]),
    ),
    (
        nt.arange(6).reshape(2, 3).to(nt.float32),
        nt.arange(12).reshape(3, 4).to(nt.float32),
        nt.matmul,
        nt.tensor([[20.0, 23.0, 26.0, 29.0], [56.0, 68.0, 80.0, 92.0]]),
    ),
    (
        nt.tensor([[1.0, 2.0, 3.0]]),
        nt.tensor([[4.0], [5.0], [6.0]]),
        nt.matmul,
        nt.tensor([[32.0]]),
    ),
    (
        nt.eye(3),
        nt.arange(9).reshape(3, 3).to(nt.float32),
        nt.matmul,
        nt.arange(9).reshape(3, 3).to(nt.float32),
    ),
    (
        nt.arange(9).reshape(3, 3).to(nt.float32),
        nt.eye(3),
        nt.matmul,
        nt.arange(9).reshape(3, 3).to(nt.float32),
    ),
    (
        nt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        nt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).T,
        nt.matmul,
        nt.tensor([[14.0, 32.0], [32.0, 77.0]]),
    ),
    (
        nt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).T,
        nt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        nt.matmul,
        nt.tensor([[17.0, 22.0, 27.0], [22.0, 29.0, 36.0], [27.0, 36.0, 45.0]]),
    ),
    (
        nt.arange(12).reshape(3, 4).to(nt.float32)[:, ::2],
        nt.ones(2, 2),
        nt.matmul,
        nt.tensor([[2.0, 2.0], [10.0, 10.0], [18.0, 18.0]]),
    ),
    (
        nt.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]).expand((8, 3, 2, 3)),
        nt.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]).expand((8, 3, 3, 2)),
        nt.matmul,
        nt.tensor([[1.0, 0.0], [1.0, 1.0]]).expand((8, 3, 2, 2)),
    ),
    (
        nt.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]).expand((3, 2, 3)),
        nt.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]).expand((8, 3, 3, 2)),
        nt.matmul,
        nt.tensor([[1.0, 0.0], [1.0, 1.0]]).expand((8, 3, 2, 2)),
    ),
    (
        nt.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]).expand((8, 3, 2, 3)),
        nt.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]).expand((8, 1, 3, 2)),
        nt.matmul,
        nt.tensor([[1.0, 0.0], [1.0, 1.0]]).expand((8, 3, 2, 2)),
    ),
    (
        nt.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).expand((8, 3, 2, 3)),
        nt.tensor([1.0, 2.0, 3.0]),
        nt.matmul,
        nt.tensor([[8.0, 26.0]]).expand((8, 3, 2)),
    ),
]


@pytest.mark.parametrize("self, other, op, result", OPS_SPEC)
def test_ops_vs_expected(self, other, op, result):
    assert op(self, other).equals(result)


def test_matmul_invalid_1d_wrongside():
    x = nt.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    y = nt.tensor([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError):
        _ = y @ x


def test_matmul_invalid_1d_wrongsize():
    x = nt.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).expand((8, 3, 2, 3))
    y = nt.tensor([1.0, 2.0, 3.0, 4.0]).expand((8, 3, 4, 4))
    with pytest.raises(RuntimeError):
        _ = x @ y


def test_matmul_dot_product():
    assert (nt.arange(5) @ nt.arange(5)).tolist() == 30.0


def test_matmul_by_value():
    torch.manual_seed(42)
    for _ in range(100):
        ndim = int(torch.randint(2, 5, (1,)).item())
        cdim = int(torch.randint(1, 10, (1,)).item())
        x1dim = int(torch.randint(1, 10, (1,)).item())
        x2dim = int(torch.randint(1, 10, (1,)).item())
        batch_dims = [int(torch.randint(1, 10, (1,)).item()) for _ in range(ndim - 2)]
        x1 = torch.rand(tuple(batch_dims + [x1dim, cdim]))
        x2 = torch.rand(tuple(batch_dims + [cdim, x2dim]))
        expected = nt.tensor(x1 @ x2)  # type: ignore
        value = nt.tensor(x1) @ nt.tensor(x2)  # type: ignore
        testing.assert_allclose(value, expected)
