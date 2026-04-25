from .core import Tensor


def add(x: Tensor, y: Tensor) -> Tensor:
    """Add two tensors component-wise, with broadcasting."""
    return x + y


def subtract(x: Tensor, y: Tensor) -> Tensor:
    """Subtract two tensors component-wise, with broadcasting."""
    return x - y


def multiply(x: Tensor, y: Tensor) -> Tensor:
    """Multiply two tensors component-wise, with broadcasting."""
    return x * y


def divide(x: Tensor, y: Tensor) -> Tensor:
    """Divide two tensors component-wise, with broadcasting."""
    return x / y


def matmul(x: Tensor, y: Tensor) -> Tensor:
    """Multiply two tensors with standard matrix multiplication."""
    return x @ y
