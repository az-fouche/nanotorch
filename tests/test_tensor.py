import pytest

import nanotorch as nt


def test_tensor_basics():
    x = nt.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert str(x) == "NtTensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])"
    assert x.shape == (2, 3)
    assert x.dim == 2
    assert x.numel == 6
    assert len(x) == 2


def test_tensor_empty():
    x = nt.tensor([])
    assert str(x) == "NtTensor([])"
    assert x.shape == (0,)
    assert x.dim == 1
    assert x.numel == 0
    assert len(x) == 0


def test_tensor_scalar():
    x = nt.tensor(0.0)
    assert str(x) == "NtTensor(0.0)"
    assert x.shape == ()
    assert x.dim == 0
    assert x.numel == 1
    assert len(x) == 0


def test_tensor_misshaped_missing_elem():
    with pytest.raises(ValueError):
        nt.tensor([[0.0, 0.0, 0.0], [1.0, 1.0]])


def test_tensor_misshaped_unhomogeneous():
    with pytest.raises(ValueError):
        nt.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, []]])


def test_tensor_misshaped_bad_type():
    with pytest.raises(ValueError):
        nt.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, "a"]])
