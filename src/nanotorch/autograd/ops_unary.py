"""Unary differentiable tensor operations and their derivative."""

from nanotorch import _C  # type: ignore[missing-import]
from nanotorch._data_type import Dtype, promote_dtypes
from nanotorch.core import Tensor, sizeof

from . import ops_spec as sp
from .function import Function
from .ops_utils import unbroadcast


class NegOp(Function):
    """Sign inversion operation.

    Parameters
    ----------
    x: Tensor
        Tensor to invert coefficients (raises if bool).

    Returns
    -------
    Tensor
        New tensor containing the coordinate-wise division.
    """

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real()),)

    def forward(self, x: Tensor) -> Tensor:
        self._shape = x.shape
        return Tensor._new_contiguous(_C.neg(x._C_view), x.shape)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (unbroadcast(-1 * grad_out, self._shape),)

    @classmethod
    def flops(cls, x: Tensor) -> int:
        return x.numel

    @classmethod
    def mem_bytes(cls, x: Tensor) -> int:
        return 2 * x.numel * sizeof(x.dtype)


class ExpOp(Function):
    """Exponentiate all tensor coefficients.

    Parameters
    ----------
    x: Tensor
        Target tensor to exponentiate.

    Returns
    -------
    Tensor
        Exponentiated tensor, cast to floating point if necessary.
    """

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real(low=-5, high=5)),)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, Dtype.Float32)
        x = x.to(dtype)
        e_x = Tensor._new_contiguous(_C.exp(x._C_view), x.shape)
        self.save_for_backward(e_x)
        return e_x

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out * self.saved_tensors[0],)

    @classmethod
    def flops(cls, x: Tensor) -> int:
        return x.numel

    @classmethod
    def mem_bytes(cls, x: Tensor) -> int:
        return 2 * x.numel * sizeof(x.dtype)


class LogOp(Function):
    """Logarithmize (Napierian) all tensor coefficients.

    Parameters
    ----------
    x: Tensor
        Target tensor to logarithmize.

    Returns
    -------
    Tensor
        Logarithmized tensor, cast to floating point if necessary.
    """

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real(low=0)),)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, Dtype.Float32)
        x = x.to(dtype)
        self.save_for_backward(x)
        return Tensor._new_contiguous(_C.log(x._C_view), x.shape)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return (grad_out * self.saved_tensors[0] ** -1,)

    @classmethod
    def flops(cls, x: Tensor) -> int:
        return x.numel

    @classmethod
    def mem_bytes(cls, x: Tensor) -> int:
        return 2 * x.numel * sizeof(x.dtype)


class SqrtOp(Function):
    """Square root of all tensor coefficients.

    Parameters
    ----------
    x: Tensor
        Target tensor to apply square root.

    Returns
    -------
    Tensor
        Sqrt tensor, cast to floating point if necessary.
    """

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real(low=0)),)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, Dtype.Float32)
        x = x.to(dtype)
        sqrt_x = Tensor._new_contiguous(_C.sqrt(x._C_view), x.shape)
        self.save_for_backward(sqrt_x)
        return sqrt_x

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        sqrt_x = self.saved_tensors[0]
        return (grad_out * 0.5 * sqrt_x**-1,)

    @classmethod
    def flops(cls, x: Tensor) -> int:
        return x.numel

    @classmethod
    def mem_bytes(cls, x: Tensor) -> int:
        return 2 * x.numel * sizeof(x.dtype)


class TanhOp(Function):
    """Tanh of all tensor coefficients.

    Parameters
    ----------
    x: Tensor
        Target tensor to apply tanh.

    Returns
    -------
    Tensor
        Tanh tensor, cast to floating point if necessary.
    """

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real()),)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, Dtype.Float32)
        x = x.to(dtype)
        tanh_x = Tensor._new_contiguous(_C.tanh(x._C_view), x.shape)
        self.save_for_backward(tanh_x)
        return tanh_x

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        tanh_x = self.saved_tensors[0]
        return (grad_out * (1 - tanh_x**2),)

    @classmethod
    def flops(cls, x: Tensor) -> int:
        return x.numel

    @classmethod
    def mem_bytes(cls, x: Tensor) -> int:
        return 2 * x.numel * sizeof(x.dtype)


class SigmoidOp(Function):
    """Sigmoid of all tensor coefficients.

    Parameters
    ----------
    x: Tensor
        Target tensor to apply sigmoid.

    Returns
    -------
    Tensor
        Sigmoid tensor, cast to floating point if necessary.
    """

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real()),)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_empty:
            return x
        dtype = promote_dtypes(x.dtype, Dtype.Float32)
        x = x.to(dtype)
        sig_x = Tensor._new_contiguous(_C.sigmoid(x._C_view), x.shape)
        self.save_for_backward(sig_x)
        return sig_x

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        sig_x = self.saved_tensors[0]
        return (grad_out * sig_x * (1 - sig_x),)

    @classmethod
    def flops(cls, x: Tensor) -> int:
        return x.numel

    @classmethod
    def mem_bytes(cls, x: Tensor) -> int:
        return 2 * x.numel * sizeof(x.dtype)


class PowOp(Function):
    """Raises all tensor coefficients to a given exponent.

    Parameters
    ----------
    x: Tensor
        Target tensor.
    exponent: Tensor | float | int | bool
        Exponent, must be scalar-like.

    Returns
    -------
    Tensor
        Resulting tensor.
    """

    op_spec = (
        sp.Input("tensor", sp.AnyShape(), sp.Real(low=0.1, high=10)),
        sp.Input("scalar", None, sp.Real(low=-5, high=5)),
    )

    def forward(self, x: Tensor, exponent: Tensor) -> Tensor:
        if x.is_empty:
            return x
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)
        dtype = promote_dtypes(x.dtype, exponent.dtype)
        exponent_fp = float(exponent.cpu().item())
        if exponent_fp < 0 and dtype <= Dtype.Int64:
            raise RuntimeError("Cannot use negative exponent with integer tensor.")
        self.save_for_backward(x, exponent)
        return Tensor._new_contiguous(_C.pow(x.to(dtype)._C_view, exponent_fp), x.shape)

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        x, exponent = self.saved_tensors
        return (grad_out * exponent * x ** (exponent - 1),)

    @classmethod
    def flops(cls, x: Tensor, exponent: Tensor) -> int:
        return x.numel

    @classmethod
    def mem_bytes(cls, x: Tensor, exponent: Tensor) -> int:
        return 2 * x.numel * sizeof(x.dtype)


class ReluOp(Function):
    """Rectified linear unit."""

    op_spec = (sp.Input("tensor", sp.AnyShape(), sp.Real()),)

    def forward(self, x: Tensor) -> Tensor:
        result = Tensor._new_contiguous(_C.relu(x._C_view), x.shape)
        self.save_for_backward(result)
        return result

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        result = self.saved_tensors[0]
        return (grad_out * (result > Tensor(0, device=result.device)),)

    @classmethod
    def flops(cls, x: Tensor) -> int:
        return x.numel

    @classmethod
    def mem_bytes(cls, x: Tensor) -> int:
        return 2 * x.numel * sizeof(x.dtype)
