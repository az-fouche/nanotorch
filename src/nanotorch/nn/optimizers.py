"""Standard parameters optimizers."""

from typing import Sequence

import nanotorch.autograd as ag
from nanotorch.core import Tensor


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, parameters: Sequence[Tensor]) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> Sequence[Tensor]:
        """Parameters tracked by the optimizer."""
        return self._parameters

    def zero_grad(self) -> None:
        """Reset all parameters gradients."""
        for p in self._parameters:
            p.zero_grad()

    def step(self) -> None:
        """Step the optimizer by updating all tracked parameters."""
        raise NotImplementedError


class GradientDescent(Optimizer):
    """Standard gradient descent.

    Parameters
    ----------
    parameters: Sequence[Tensor]
        Model parameters.
    lr: float
        Learning rate.
    """

    def __init__(self, parameters: Sequence[Tensor], lr: float) -> None:
        super().__init__(parameters)
        self._lr = lr

    def step(self) -> None:
        for p in self.parameters:
            if p.requires_grad:
                if p.grad is None:
                    raise RuntimeError("Gradients not set, did you call backward?")
                with ag.no_grad():
                    p -= p.grad * self._lr
