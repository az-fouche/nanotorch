import nanotorch as nt


def assert_allclose(x1: nt.Tensor, x2: nt.Tensor, tol: float = 1e-6) -> None:
    if x1.shape != x2.shape:
        raise AssertionError("Shapes do not match!")
    x1f, x2f = x1.flatten().tolist(), x2.flatten().tolist()
    if not isinstance(x1f, list):
        x1f = [x1f]
    if not isinstance(x2f, list):
        x2f = [x2f]
    for i, j in zip(x1f, x2f):
        if abs(i - j) > tol:
            raise AssertionError(f"Tensors do not match ({i} != {j}).")
