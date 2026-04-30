"""Neural network parametrized modules."""

from typing import Any, Sequence

from nanotorch.autograd import ReluOp
from nanotorch.core import Tensor
from nanotorch.factories import rand, zeros


class Module:
    """Base class for parametrized modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> Sequence[Tensor]:
        """Sequence of all module parameters."""
        return []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class Linear(Module):
    """Linear module xW + b.

    Parameters
    ----------
    fan_in: int
        Input dimension (W.shape[0]).
    fan_out: int
        Output dimension (W.shape[1] and b.shape[0]).
    bias: bool
        Use a `b` term.
    """

    def __init__(self, fan_in: int, fan_out: int, bias: bool = True) -> None:
        self._W = rand(fan_in, fan_out) - 0.5
        self._W.enable_grad()
        if bias:
            self._b = zeros(fan_out, requires_grad=True)
        else:
            self._b = None

    def parameters(self) -> Sequence[Tensor]:
        if self._b is None:
            return [self._W]
        return [self._W, self._b]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise NotImplementedError(
                f"Can only forward a 2D tensor for now, got {x.ndim}"
            )
        if x.shape[1] != self._W.shape[0]:
            raise ValueError(
                f"Cannot align axes of size {x.shape[1]} and {self._W.shape[0]}."
            )
        z = x @ self._W
        if self._b is not None:
            z = z + self._b
        return z


class ReLU(Module):
    """Rectified linear unit layer."""

    def forward(self, x: Tensor) -> Tensor:
        return ReluOp.apply(x)


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self._modules = modules

    def parameters(self) -> Sequence[Tensor]:
        parameters: list[Tensor] = []
        for mod in self._modules:
            parameters.extend(mod.parameters())
        return parameters

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        x: None
        for i, mod in enumerate(self._modules):
            if i == 0:
                x = mod(*args, **kwargs)
            else:
                x = mod(x)
        return x
