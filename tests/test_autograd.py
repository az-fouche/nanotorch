import math
import random

import pytest

import nanotorch as nt
import nanotorch.autograd as ag
from nanotorch import testing
from nanotorch.autograd import ops_spec as sp
from nanotorch.core import Tensor

nt.manual_seed(42)

# Grad state tests


def test_requires_grad_default_true():
    x = nt.tensor(0.0)
    assert not x.requires_grad


def test_requires_grad_enabled():
    x = nt.tensor(0.0, requires_grad=True)
    assert x.requires_grad
    assert x.grad is None
    assert x.grad_fn is None


def test_requires_grad_int_raises():
    with pytest.raises(RuntimeError):
        nt.tensor(1, dtype=nt.int64, requires_grad=True)


def test_backward_oneleaf_onestep():
    x = nt.tensor(1.0, requires_grad=True)
    loss = 3 * x
    loss.backward()
    assert loss.grad is None
    assert x.grad is not None
    assert x.grad.tolist() == 3.0


def test_backward_oneleaf_twosteps():
    x = nt.tensor(1.0, requires_grad=True)
    y = 3 * x
    loss = y.exp()
    loss.backward()
    assert loss.grad is None
    assert y.grad is None
    assert x.grad is not None
    testing.assert_allclose(x.grad, nt.tensor(60.2566), tol=0.001)


def test_backward_twoleaf_twosteps():
    x1 = nt.tensor(1.0, requires_grad=True)
    x2 = nt.tensor(5.0, requires_grad=True)
    y = x1 * x2
    loss = y.log()
    loss.backward()
    assert loss.grad is None
    assert y.grad is None
    assert x1.grad is not None
    assert x2.grad is not None
    testing.assert_allclose(x1.grad, nt.tensor(1.0))
    testing.assert_allclose(x2.grad, nt.tensor(0.2))


def test_backward_twoleaf_twosteps_1d():
    x1 = nt.tensor([1.0, 5.0], requires_grad=True)
    x2 = nt.tensor([5.0, 1.0], requires_grad=True)
    y = x1 * x2
    z = y.log()
    loss = z.sum()
    loss.backward()
    assert loss.grad is None
    assert z.grad is None
    assert y.grad is None
    assert x1.grad is not None
    assert x2.grad is not None
    testing.assert_allclose(x1.grad, nt.tensor([1.0, 0.2]))
    testing.assert_allclose(x2.grad, nt.tensor([0.2, 1.0]))


def test_backward_sum_axis():
    x = nt.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
    y = x.sum(axis=0)
    loss = y.sum()
    loss.backward()
    assert loss.grad is None
    assert y.grad is None
    assert x.grad is not None
    testing.assert_allclose(x.grad, nt.tensor([[1.0, 1.0], [1.0, 1.0]]))


def test_backward_sum_keepdim():
    x = nt.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
    y = x.sum(axis=0, keepdim=True)
    loss = y.sum()
    loss.backward()
    assert loss.grad is None
    assert y.grad is None
    assert x.grad is not None
    testing.assert_allclose(x.grad, nt.tensor([[1.0, 1.0], [1.0, 1.0]]))


def test_backward_transpose():
    x = nt.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    g = nt.arange(6).reshape(3, 2)
    loss = (x.T * g).sum()
    loss.backward()
    assert loss.grad is None
    assert g.grad is None
    assert x.grad is not None
    testing.assert_allclose(x.grad, g.T)


def test_backward_inplace_ops_leaf_raises():
    x2 = nt.tensor([5.0, 1.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        x2 *= 1.5


def test_backward_inplace_ops_guard_raises():
    x = nt.tensor([5.0, 1.0], requires_grad=True)
    y = x.exp()
    y += 3
    loss = y.sum()
    with pytest.raises(RuntimeError):
        loss.backward()


def test_backward_order_unbalanced():
    x = nt.tensor([1.0, 2.0], requires_grad=True)
    a = x.exp()
    b = a * 2
    c = a * 3
    loss = (b + c).sum()
    loss.backward()
    assert loss.grad is None
    assert c.grad is None
    assert b.grad is None
    assert a.grad is None
    assert x.grad is not None
    testing.assert_allclose(x.grad, nt.tensor([5 * math.e, 5 * math.e**2]))


def test_version_check_on_backward():
    x = nt.tensor([1.0, 2.0], requires_grad=True)
    y = 2 * x
    z = 3 * y
    y += 2
    with pytest.raises(RuntimeError):
        z.backward()


def test_no_grad():
    x = nt.tensor([1.0, 2.0], requires_grad=True)
    with nt.no_grad():
        y = 2 * x
    assert y.grad_fn is None


# Autograd engine autocheck


@pytest.mark.parametrize("op", ag.ALL_OPS_)
def test_ops_specs(op: type[ag.Function]):
    nt.manual_seed(42)
    random.seed(42)
    for _ in range(5):
        inputs = sp.gen_random_input_for(op.op_spec, max_dim=5, size_factor=1)
        tensor_in = [
            x.to(nt.float64) if x.dtype == nt.float32 else x
            for x in inputs
            if isinstance(x, Tensor)
        ]
        extra_args = [x for x in inputs if not isinstance(x, Tensor)]
        testing.gradcheck(op, *tensor_in, extra_args=extra_args)
