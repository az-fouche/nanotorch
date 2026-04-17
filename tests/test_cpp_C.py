from array import array

from nanotorch import _C  # type: ignore


def test_cpp_sum_f():
    x = array("f", [1.0, 2.0, 3.0])
    assert _C.sum(x) == 6.0


def test_cpp_sum_d():
    x = array("d", [1.0, 2.0, 3.0])
    assert _C.sum(x) == 6.0


def test_cpp_sum_q():
    x = array("q", [1, 2, 3])
    assert _C.sum(x) == 6.0


def test_cpp_sum_i():
    x = array("i", [1, 2, 3])
    assert _C.sum(x) == 6.0


def test_cpp_sum_b():
    x = array("b", [True, False, True])
    assert _C.sum(x) == 2.0
