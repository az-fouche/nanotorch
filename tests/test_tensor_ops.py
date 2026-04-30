import pytest

import nanotorch as nt


@pytest.mark.parametrize(
    "self, other, op, result",
    [
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
        (nt.zeros(5, 7), nt.full(5, 7, value=3.14), nt.subtract, nt.full(5, 7, value=-3.14)),
        (
            nt.zeros(5, 7, dtype=nt.bool_),
            nt.full(5, 7, value=3.14),
            nt.subtract,
            nt.full(5, 7, value=-3.14),
        ),
        (nt.zeros(5, 7), nt.full(5, 1, value=3.14), nt.subtract, nt.full(5, 7, value=-3.14)),
        (nt.zeros(5, 7), nt.full(7, value=3.14), nt.subtract, nt.full(5, 7, value=-3.14)),
        (nt.zeros(7, 5).T, nt.full(7, value=3.14), nt.subtract, nt.full(5, 7, value=-3.14)),
        # multiply
        (nt.tensor(1.0), 3.14, nt.multiply, nt.tensor(3.14)),
        (3.14, nt.tensor(1.0), nt.multiply, nt.tensor(3.14)),
        (nt.tensor(1.0), -3.14, nt.multiply, nt.tensor(-3.14)),
        (nt.tensor(1.0), True, nt.multiply, nt.tensor(1.0)),
        (nt.tensor(True), 3.14, nt.multiply, nt.tensor(3.14)),
        (nt.ones(5), 3.14, nt.multiply, nt.full(5, value=3.14)),
        (nt.ones(5, 7), 3.14, nt.multiply, nt.full(5, 7, value=3.14)),
        (nt.ones(5, 7), nt.full(5, 7, value=3.14), nt.multiply, nt.full(5, 7, value=3.14)),
        (
            nt.ones(5, 7, dtype=nt.bool_),
            nt.full(5, 7, value=3.14),
            nt.multiply,
            nt.full(5, 7, value=3.14),
        ),
        (nt.ones(5, 7), nt.full(5, 1, value=3.14), nt.multiply, nt.full(5, 7, value=3.14)),
        (nt.ones(5, 7), nt.full(7, value=3.14), nt.multiply, nt.full(5, 7, value=3.14)),
        (nt.ones(7, 5).T, nt.full(7, value=3.14), nt.multiply, nt.full(5, 7, value=3.14)),
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
            nt.tensor(
                [[17.0, 22.0, 27.0], [22.0, 29.0, 36.0], [27.0, 36.0, 45.0]]
            ),
        ),
        (
            nt.arange(12).reshape(3, 4).to(nt.float32)[:, ::2],
            nt.ones(2, 2),
            nt.matmul,
            nt.tensor([[2.0, 2.0], [10.0, 10.0], [18.0, 18.0]]),
        ),
    ],
)
def test_ops_vs_expected(self, other, op, result):
    assert op(self, other).equals(result)
