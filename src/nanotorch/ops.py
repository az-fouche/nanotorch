from .core import Tensor


def add(x: Tensor, y: Tensor) -> Tensor:
    return x + y


def subtract(x: Tensor, y: Tensor) -> Tensor:
    return x - y


def multiply(x: Tensor, y: Tensor) -> Tensor:
    return x * y


def divide(x: Tensor, y: Tensor) -> Tensor:
    return x / y
