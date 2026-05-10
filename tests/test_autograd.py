import math
import random

import pytest

import nanotorch as nt
import nanotorch.autograd as ag
from nanotorch import testing
from nanotorch.autograd import ops_spec as sp
from nanotorch.core import Tensor, TensorLike

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


@pytest.mark.parametrize(
    "op",
    [
        ag.AddOp,
        ag.ExpOp,
        ag.ExpandOp,
        ag.LogOp,
        ag.MatmulOp,
        ag.MeanOp,
        ag.MulOp,
        ag.NegOp,
        ag.PowOp,
        ag.ReluOp,
        ag.ReshapeOp,
        ag.SigmoidOp,
        ag.SqrtOp,
        ag.SubOp,
        ag.SumOp,
        ag.TOp,
        ag.TanhOp,
        ag.TransposeOp,
        ag.TrueDivOp,
        ag.MinOp,
        ag.MaxOp,
    ],
)
def test_ops_specs(op: type[ag.Function]):
    nt.manual_seed(42)
    random.seed(42)
    for _ in range(5):
        inputs = []
        for input_ in op.op_spec:
            domain = input_.domain
            if isinstance(domain, (sp.Bool, sp.Real)):
                inputs.append(_gen_float_like(input_, inputs))
            else:
                inputs.extend(_gen_axis_like(domain, inputs))
        tensor_in = [x for x in inputs if isinstance(x, Tensor)]
        extra_args = [x for x in inputs if not isinstance(x, Tensor)]
        testing.gradcheck(op, *tensor_in, extra_args=extra_args)


def _gen_float_like(input_: sp.Input, inputs: list[TensorLike]) -> TensorLike:
    if input_.kind == "scalar":
        assert input_.shape is None
        shape = ()
    elif isinstance(input_.shape, sp.AnyShape):
        ndim = random.randint(input_.shape.min_ndim, 5)
        shape = nt.randint(1, 6, (ndim,)).tolist()
    elif isinstance(input_.shape, sp.BroadcastableTo):
        ref = inputs[input_.shape.ref]
        assert isinstance(ref, Tensor)
        ndim = random.randint(0, ref.ndim + 1)
        suffix = ref.shape[ref.ndim - ndim :]
        shape = [1 if random.random() < 0.5 else ax for ax in suffix]
    elif isinstance(input_.shape, sp.MatmulBroadcast):
        ref = inputs[input_.shape.ref]
        assert isinstance(ref, Tensor)
        shape = [1 if random.random() < 0.5 else ax for ax in ref.shape[:-2]]
        shape += [ref.shape[-1], random.randint(1, 8)]
    else:
        raise AssertionError(f"Unhandled shape spec: {input_.shape}")
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, list):
        shape = tuple(shape)
    assert isinstance(shape, tuple)

    if isinstance(input_.domain, sp.Real):
        dtype = nt.float64
    elif isinstance(input_.domain, sp.Bool):
        dtype = nt.bool_
    else:
        raise ValueError(f"Unrecognized type {input_.domain}")
    x = nt.rand(*shape, dtype=nt.float64, requires_grad=False).to(dtype)

    if isinstance(input_.domain, sp.Real):
        x = (input_.domain.high - input_.domain.low) * x + input_.domain.low
    if input_.kind == "tensor":
        x = x.clone()
        x.enable_grad()
    else:
        x = x.item()
    return x


def _gen_axis_like(
    domain: sp.InputDomain, inputs: list[TensorLike]
) -> tuple[int, ...] | list[tuple[int, ...]]:
    ref = inputs[domain.ref]  # type: ignore
    assert isinstance(ref, Tensor)
    if isinstance(domain, (sp.Axis, sp.AxisSet)):
        naxes = (
            1 if isinstance(domain, (sp.Axis)) else random.randint(domain.min, ref.ndim)
        )
        axes = tuple(random.sample(range(ref.ndim), naxes))
        if isinstance(domain, sp.AxisSet) and not domain.split:
            return [axes]
        return axes
    elif isinstance(domain, sp.AxisPermutation):
        return tuple(random.sample(range(ref.ndim), ref.ndim))
    elif isinstance(domain, sp.AxisReshape):
        return _gen_reshape(ref)
    elif isinstance(domain, sp.AxisExpand):
        return _gen_expand(ref)
    else:
        raise AssertionError(f"Cannot handle domain {domain}")


def _gen_reshape(ref: Tensor) -> tuple[int, ...]:
    factors, n = [], math.prod(ref.shape)
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1
    if n > 1:
        factors.append(n)
    ndim: int = 1 + random.randint(0, len(factors))
    buckets = [1] * ndim
    for f in factors:
        buckets[random.randint(0, len(buckets) - 1)] *= f
    buckets += [1] * random.randint(0, 5)
    random.shuffle(buckets)
    return tuple(buckets)  # type: ignore


def _gen_expand(ref: Tensor) -> tuple[int, ...]:
    shape = []
    for ax in ref.shape[::-1]:
        shape = [random.randint(1, 8) if ax == 1 else ax] + shape
    while True:
        shape = [random.randint(1, 8)] + shape
        if random.random() < 0.5:
            break
    return tuple(shape)
