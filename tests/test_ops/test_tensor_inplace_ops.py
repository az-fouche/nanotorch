import pytest
from conftest import requires_cuda

import nanotorch as nt
from nanotorch import _C, testing

# ---------- Python-level inplace operators ----------


def _iadd(a, b):
    a += b
    return a


def _isub(a, b):
    a -= b
    return a


def _imul(a, b):
    a *= b
    return a


def _itruediv(a, b):
    a /= b
    return a


# Each entry: (lhs_factory, rhs, op, expected). lhs_factory is a callable so
# parametrize state stays clean across runs (inplace mutates the lhs).
OPS_SPEC = [
    # iadd
    (lambda: nt.tensor(0.0), 3.14, _iadd, nt.tensor(3.14)),
    (lambda: nt.tensor(0.0), -3.14, _iadd, nt.tensor(-3.14)),
    (lambda: nt.tensor(0.0), True, _iadd, nt.tensor(1.0)),
    (lambda: nt.zeros(5), 3.14, _iadd, nt.full(5, value=3.14)),
    (lambda: nt.zeros(5, 7), 3.14, _iadd, nt.full(5, 7, value=3.14)),
    (
        lambda: nt.zeros(5, 7),
        nt.full(5, 7, value=3.14),
        _iadd,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.zeros(5, 7),
        nt.full(5, 1, value=3.14),
        _iadd,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.zeros(5, 7),
        nt.full(7, value=3.14),
        _iadd,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.zeros(7, 5).T,
        nt.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        _iadd,
        nt.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 5),
    ),
    # isub
    (lambda: nt.tensor(0.0), 3.14, _isub, nt.tensor(-3.14)),
    (lambda: nt.tensor(0.0), -3.14, _isub, nt.tensor(3.14)),
    (lambda: nt.zeros(5), 3.14, _isub, nt.full(5, value=-3.14)),
    (lambda: nt.zeros(5, 7), 3.14, _isub, nt.full(5, 7, value=-3.14)),
    (
        lambda: nt.zeros(5, 7),
        nt.full(5, 7, value=3.14),
        _isub,
        nt.full(5, 7, value=-3.14),
    ),
    (
        lambda: nt.zeros(5, 7),
        nt.full(5, 1, value=3.14),
        _isub,
        nt.full(5, 7, value=-3.14),
    ),
    (
        lambda: nt.zeros(5, 7),
        nt.full(7, value=3.14),
        _isub,
        nt.full(5, 7, value=-3.14),
    ),
    (
        lambda: nt.zeros(7, 5).T,
        nt.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        _isub,
        nt.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]] * 5),
    ),
    # imul
    (lambda: nt.tensor(1.0), 3.14, _imul, nt.tensor(3.14)),
    (lambda: nt.tensor(1.0), -3.14, _imul, nt.tensor(-3.14)),
    (lambda: nt.tensor(1.0), True, _imul, nt.tensor(1.0)),
    (lambda: nt.ones(5), 3.14, _imul, nt.full(5, value=3.14)),
    (lambda: nt.ones(5, 7), 3.14, _imul, nt.full(5, 7, value=3.14)),
    (
        lambda: nt.ones(5, 7),
        nt.full(5, 7, value=3.14),
        _imul,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.ones(5, 7),
        nt.full(5, 1, value=3.14),
        _imul,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.ones(5, 7),
        nt.full(7, value=3.14),
        _imul,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.ones(7, 5).T,
        nt.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        _imul,
        nt.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 5),
    ),
    # itruediv
    (lambda: nt.tensor(6.28), 2.0, _itruediv, nt.tensor(3.14)),
    (lambda: nt.tensor(6.28), -2.0, _itruediv, nt.tensor(-3.14)),
    (lambda: nt.full(5, value=6.28), 2.0, _itruediv, nt.full(5, value=3.14)),
    (
        lambda: nt.full(5, 7, value=6.28),
        2.0,
        _itruediv,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.full(5, 7, value=6.28),
        nt.full(5, 7, value=2.0),
        _itruediv,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.full(5, 7, value=6.28),
        nt.full(5, 1, value=2.0),
        _itruediv,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.full(5, 7, value=6.28),
        nt.full(7, value=2.0),
        _itruediv,
        nt.full(5, 7, value=3.14),
    ),
    (
        lambda: nt.full(7, 5, value=24.0).T,
        nt.tensor([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]),
        _itruediv,
        nt.tensor([[24.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0]] * 5),
    ),
]


@pytest.mark.parametrize("lhs_factory, rhs, op, expected", OPS_SPEC)
def test_inplace_ops_vs_expected(lhs_factory, rhs, op, expected):
    lhs = lhs_factory()
    op(lhs, rhs)
    assert lhs.equals(expected)


@requires_cuda
@pytest.mark.parametrize("lhs_factory, rhs, op, expected", OPS_SPEC)
def test_inplace_ops_vs_expected_cuda(lhs_factory, rhs, op, expected):
    lhs = lhs_factory().to("cuda")
    if isinstance(rhs, nt.Tensor):
        rhs = rhs.to("cuda")
    op(lhs, rhs)
    assert lhs.cpu().equals(expected)


# ---------- Non-contiguous LHS on CUDA (must keep the view on-device) ----------


@requires_cuda
def test_iadd_transpose_lhs_on_cuda():
    a = nt.zeros(7, 5).to("cuda").T
    a += nt.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to("cuda")
    expected = nt.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 5)
    assert a.cpu().equals(expected)


@requires_cuda
def test_imul_transpose_lhs_on_cuda():
    a = nt.ones(7, 5).to("cuda").T
    a *= nt.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to("cuda")
    expected = nt.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 5)
    assert a.cpu().equals(expected)


# ---------- Storage identity, version bump, return value ----------


@pytest.mark.parametrize("op", [_iadd, _isub, _imul, _itruediv])
def test_inplace_preserves_storage(op):
    t = nt.full(5, value=2.0)
    s = t.storage
    op(t, 1.0)
    assert t.storage is s


@pytest.mark.parametrize("op", [_iadd, _isub, _imul, _itruediv])
def test_inplace_bumps_version(op):
    t = nt.full(5, value=2.0)
    v0 = t.version
    op(t, 1.0)
    assert t.version == v0 + 1
    op(t, 1.0)
    assert t.version == v0 + 2


@pytest.mark.parametrize("op", [_iadd, _isub, _imul, _itruediv])
def test_inplace_returns_aliases_same_storage(op):
    t = nt.full(5, value=2.0)
    s = t.storage
    out = op(t, 1.0)
    assert out.storage is s
    assert out.shape == t.shape


# ---------- Mutation through views ----------


def test_iadd_through_slice_view_mutates_base():
    a = nt.zeros(4, 4)
    a[1] += 1.0
    assert a[1].equals(nt.full(4, value=1.0))
    assert a[0].equals(nt.zeros(4))
    assert a[2].equals(nt.zeros(4))
    assert a[3].equals(nt.zeros(4))


def test_iadd_through_transpose_view_mutates_base():
    a = nt.zeros(3, 4)
    t = a.T
    t += nt.tensor([0.0, 1.0, 2.0])
    expected = nt.tensor([[0.0] * 4, [1.0] * 4, [2.0] * 4])
    assert a.equals(expected)


def test_iadd_through_expand_view_mutates_base():
    a = nt.zeros(3, 4)
    a[:, 1:3] += 1.0
    expected = nt.tensor(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    assert a.equals(expected)


# ---------- Self-aliased operands ----------


def test_iadd_self_aliased():
    t = nt.full(5, value=2.0)
    t += t
    assert t.equals(nt.full(5, value=4.0))


def test_imul_self_aliased():
    t = nt.full(5, value=3.0)
    t *= t
    assert t.equals(nt.full(5, value=9.0))


def test_isub_self_aliased():
    t = nt.full(5, value=3.0)
    t -= t
    assert t.equals(nt.zeros(5))


def test_itruediv_self_aliased():
    t = nt.full(5, value=3.0)
    t /= t
    assert t.equals(nt.ones(5))


# ---------- Broadcast / shape errors ----------


def test_iadd_broadcast_failure_lhs_too_few_dims():
    t = nt.zeros(4)
    with pytest.raises(ValueError):
        t += nt.zeros(3, 4)


def test_iadd_broadcast_failure_dim_mismatch():
    t = nt.zeros(3, 4)
    with pytest.raises(ValueError):
        t += nt.zeros(5)


# ---------- C++ primitives (strict shape/dtype/device, no broadcast) ----------


CPP_INPLACE_OPS = [
    ("add_inplace", _C.add_inplace),
    ("sub_inplace", _C.sub_inplace),
    ("mul_inplace", _C.mul_inplace),
    ("div_inplace", _C.div_inplace),
    ("copy_inplace", _C.copy_inplace),
]


@pytest.mark.parametrize("name, op", CPP_INPLACE_OPS)
def test_cpp_inplace_shape_mismatch_raises(name, op):
    a = nt.full(5, 7, value=2.0)
    b = nt.full(5, value=2.0)
    with pytest.raises(ValueError):
        op(a._C_view, b._C_view)


@pytest.mark.parametrize("name, op", CPP_INPLACE_OPS)
def test_cpp_inplace_dtype_mismatch_raises(name, op):
    a = nt.zeros(5, dtype=nt.float32)
    b = nt.zeros(5, dtype=nt.float64)
    with pytest.raises(ValueError):
        op(a._C_view, b._C_view)


@requires_cuda
@pytest.mark.parametrize("name, op", CPP_INPLACE_OPS)
def test_cpp_inplace_device_mismatch_raises(name, op):
    a = nt.zeros(5)
    b = nt.zeros(5).to("cuda")
    with pytest.raises(ValueError):
        op(a._C_view, b._C_view)


def test_cpp_div_inplace_bool_dtype_raises():
    a = nt.full(5, value=True, dtype=nt.bool_)
    b = nt.full(5, value=True, dtype=nt.bool_)
    with pytest.raises(ValueError):
        _C.div_inplace(a._C_view, b._C_view)


@pytest.mark.parametrize("name, op", CPP_INPLACE_OPS)
def test_cpp_inplace_bumps_version(name, op):
    a = nt.full(5, value=2.0)
    b = nt.full(5, value=1.0)
    v0 = a.version
    op(a._C_view, b._C_view)
    assert a.version == v0 + 1


def test_cpp_add_inplace_correctness():
    a = nt.tensor([1.0, 2.0, 3.0])
    b = nt.tensor([10.0, 20.0, 30.0])
    _C.add_inplace(a._C_view, b._C_view)
    assert a.equals(nt.tensor([11.0, 22.0, 33.0]))


def test_cpp_sub_inplace_correctness():
    a = nt.tensor([10.0, 20.0, 30.0])
    b = nt.tensor([1.0, 2.0, 3.0])
    _C.sub_inplace(a._C_view, b._C_view)
    assert a.equals(nt.tensor([9.0, 18.0, 27.0]))


def test_cpp_mul_inplace_correctness():
    a = nt.tensor([1.0, 2.0, 3.0])
    b = nt.tensor([10.0, 20.0, 30.0])
    _C.mul_inplace(a._C_view, b._C_view)
    assert a.equals(nt.tensor([10.0, 40.0, 90.0]))


def test_cpp_div_inplace_correctness():
    a = nt.tensor([10.0, 20.0, 30.0])
    b = nt.tensor([2.0, 4.0, 6.0])
    _C.div_inplace(a._C_view, b._C_view)
    assert a.equals(nt.tensor([5.0, 5.0, 5.0]))


def test_cpp_copy_inplace_overwrites():
    a = nt.zeros(3, 4)
    b = nt.full(3, 4, value=5.0)
    _C.copy_inplace(a._C_view, b._C_view)
    assert a.equals(b)


def test_cpp_copy_inplace_through_view():
    a = nt.zeros(3, 4)
    src = nt.tensor([1.0, 2.0, 3.0, 4.0])
    _C.copy_inplace(a[1]._C_view, src._C_view)
    expected = nt.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert a.equals(expected)


# ---------- Autograd guards ----------


@pytest.mark.parametrize("op", [_iadd, _isub, _imul, _itruediv])
def test_leaf_inplace_with_grad_raises(op):
    x = nt.tensor([1.0, 2.0, 3.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        op(x, 1.5)


def test_inplace_invalidates_saved_tensor():
    x = nt.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.exp()
    y += 1
    loss = y.sum()
    with pytest.raises(RuntimeError):
        loss.backward()


def test_inplace_does_not_invalidate_when_nothing_saved():
    x = nt.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x + 1
    y += 2
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    testing.assert_allclose(x.grad, nt.tensor([1.0, 1.0, 1.0]))
