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
        (nt.zeros(5), 3.14, nt.add, nt.full(5, 3.14)),
        (nt.zeros((5, 7)), 3.14, nt.add, nt.full((5, 7), 3.14)),
        (nt.zeros((5, 7)), nt.full((5, 7), 3.14), nt.add, nt.full((5, 7), 3.14)),
        (
            nt.zeros((5, 7), dtype=nt.bool_),
            nt.full((5, 7), 3.14),
            nt.add,
            nt.full((5, 7), 3.14),
        ),
        (nt.zeros((5, 7)), nt.full((5, 1), 3.14), nt.add, nt.full((5, 7), 3.14)),
        (nt.zeros((5, 7)), nt.full((7,), 3.14), nt.add, nt.full((5, 7), 3.14)),
        (nt.zeros((7, 5)).T, nt.full((7,), 3.14), nt.add, nt.full((5, 7), 3.14)),
        # subtract
        (nt.tensor(0.0), 3.14, nt.subtract, nt.tensor(-3.14)),
        (3.14, nt.tensor(0.0), nt.subtract, nt.tensor(3.14)),
        (nt.tensor(0.0), -3.14, nt.subtract, nt.tensor(3.14)),
        (nt.tensor(0.0), True, nt.subtract, nt.tensor(-1.0)),
        (nt.tensor(False), 3.14, nt.subtract, nt.tensor(-3.14)),
        (nt.zeros(5), 3.14, nt.subtract, nt.full(5, -3.14)),
        (nt.zeros((5, 7)), 3.14, nt.subtract, nt.full((5, 7), -3.14)),
        (nt.zeros((5, 7)), nt.full((5, 7), 3.14), nt.subtract, nt.full((5, 7), -3.14)),
        (
            nt.zeros((5, 7), dtype=nt.bool_),
            nt.full((5, 7), 3.14),
            nt.subtract,
            nt.full((5, 7), -3.14),
        ),
        (nt.zeros((5, 7)), nt.full((5, 1), 3.14), nt.subtract, nt.full((5, 7), -3.14)),
        (nt.zeros((5, 7)), nt.full((7,), 3.14), nt.subtract, nt.full((5, 7), -3.14)),
        (nt.zeros((7, 5)).T, nt.full((7,), 3.14), nt.subtract, nt.full((5, 7), -3.14)),
        # multiply
        (nt.tensor(1.0), 3.14, nt.multiply, nt.tensor(3.14)),
        (3.14, nt.tensor(1.0), nt.multiply, nt.tensor(3.14)),
        (nt.tensor(1.0), -3.14, nt.multiply, nt.tensor(-3.14)),
        (nt.tensor(1.0), True, nt.multiply, nt.tensor(1.0)),
        (nt.tensor(True), 3.14, nt.multiply, nt.tensor(3.14)),
        (nt.ones(5), 3.14, nt.multiply, nt.full(5, 3.14)),
        (nt.ones((5, 7)), 3.14, nt.multiply, nt.full((5, 7), 3.14)),
        (nt.ones((5, 7)), nt.full((5, 7), 3.14), nt.multiply, nt.full((5, 7), 3.14)),
        (
            nt.ones((5, 7), dtype=nt.bool_),
            nt.full((5, 7), 3.14),
            nt.multiply,
            nt.full((5, 7), 3.14),
        ),
        (nt.ones((5, 7)), nt.full((5, 1), 3.14), nt.multiply, nt.full((5, 7), 3.14)),
        (nt.ones((5, 7)), nt.full((7,), 3.14), nt.multiply, nt.full((5, 7), 3.14)),
        (nt.ones((7, 5)).T, nt.full((7,), 3.14), nt.multiply, nt.full((5, 7), 3.14)),
        # divide
        (nt.tensor(6.28), 2.0, nt.divide, nt.tensor(3.14)),
        (6.28, nt.tensor(2.0), nt.divide, nt.tensor(3.14)),
        (nt.tensor(6.28), -2.0, nt.divide, nt.tensor(-3.14)),
        (nt.tensor(3.14), True, nt.divide, nt.tensor(3.14)),
        (nt.tensor(False), 2.0, nt.divide, nt.tensor(0.0)),
        (nt.full(5, 6.28), 2.0, nt.divide, nt.full(5, 3.14)),
        (nt.full((5, 7), 6.28), 2.0, nt.divide, nt.full((5, 7), 3.14)),
        (
            nt.full((5, 7), 6.28),
            nt.full((5, 7), 2.0),
            nt.divide,
            nt.full((5, 7), 3.14),
        ),
        (
            nt.zeros((5, 7), dtype=nt.bool_),
            nt.full((5, 7), 2.0),
            nt.divide,
            nt.zeros((5, 7)),
        ),
        (
            nt.full((5, 7), 6.28),
            nt.full((5, 1), 2.0),
            nt.divide,
            nt.full((5, 7), 3.14),
        ),
        (
            nt.full((5, 7), 6.28),
            nt.full((7,), 2.0),
            nt.divide,
            nt.full((5, 7), 3.14),
        ),
        (
            nt.full((7, 5), 6.28).T,
            nt.full((7,), 2.0),
            nt.divide,
            nt.full((5, 7), 3.14),
        ),
    ],
)
def test_ops_vs_expected(self, other, op, result):
    assert op(self, other).equals(result)
