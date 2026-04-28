from typing import Any

from nanotorch.core import Tensor, TensorLike


class Function:
    """Differentiable function."""

    def __init__(self):
        self._saved_tensors: tuple[TensorLike, ...] | None = None

    def __repr__(self) -> str:
        inner = (
            ""
            if self.saved_tensors is None
            else ", ".join("x" + str(i) for i in range(len(self.saved_tensors)))
        )
        return f"[{self.__class__.__name__}({inner})]"

    @property
    def saved_tensors(self) -> tuple[TensorLike, ...] | None:
        """Returns tensors saved during forward."""
        return self._saved_tensors

    @classmethod
    def apply(cls, *inputs: Any) -> Tensor:
        """Runs forward and creates ops/tensor bindings."""
        self = cls()
        output = self.forward(*inputs)
        if any(t.requires_grad for t in inputs if isinstance(t, Tensor)):
            output.attach_grad_fn(self)
        return output

    def save_for_backward(self, *tensors: TensorLike) -> None:
        """Registers tensors for backward op."""
        self._saved_tensors = tuple(tensors)

    def forward(self, *inputs: Any) -> Tensor:
        """Wraps ops forward."""
        raise NotImplementedError

    def backward(self, grad_out: Tensor) -> tuple[Tensor]:
        """Defines ops backward."""
        raise NotImplementedError
