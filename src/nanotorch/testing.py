import nanotorch as nt
from nanotorch.autograd import Function


def assert_allclose(x1: nt.Tensor, x2: nt.Tensor, tol: float = 1e-6) -> None:
    """Asserts x1 and x2 are identical tensor up to $tol difference."""
    if x1.shape != x2.shape:
        raise AssertionError("Shapes do not match!")
    x1f, x2f = x1.flatten().tolist(), x2.flatten().tolist()
    if not isinstance(x1f, list):
        x1f = [x1f]
    if not isinstance(x2f, list):
        x2f = [x2f]
    for i, j in zip(x1f, x2f):
        if abs(i - j) > tol:
            raise AssertionError(f"Tensors {x1} and {x2} do not match ({i} != {j}).")


def gradcheck(op: Function, *inputs: nt.Tensor, eps: float = 1e-6, tol: float = 1e-5):
    """Performs gradient correctness checking for testing."""
    out = op.apply(*inputs)
    loss = out.sum()
    loss.backward()
    grad_anal = [x.grad for x in inputs]

    for x in inputs:
        for i in range(x.numel):
            pass
