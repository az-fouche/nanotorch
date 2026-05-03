"""Base tensor class and core operations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Callable, Sequence

from nanotorch import _C

from ._data_type import (
    Dtype,
    InputType,
    dtype_from_type,
    is_bool,
    is_float,
    is_int,
    promote_dtypes,
)
from ._device import Device, DeviceLiteral, get_std_device
from ._indexing import (
    TensorShape,
    broadcast_shapes,
    expand_ellipsis,
    is_contiguous_view,
)

if TYPE_CHECKING:
    from nanotorch.autograd import Function

MAX_TENSOR_LEVEL = 32


class Tensor:
    """Base tensor class.

    Nanotorch tensors are typed n-dimensional arrays with native autograd
    support and core operations.

    Parameters
    ----------
    data: list | float | int | bool | ArrayLike
        Tensor coefficients, can be scalar or list-like, provided as python
        native types. Also supports any array-like structure from torch/numpy/jax/pandas.
        Only numerical data formats are supported.
    dtype: Dtype
        Data type to cast the values to. If not set, the tensor will be promoted
        to the most precise data type following the DataTypes enum order.
    requires_grad: bool
        Enables autograd for this tensor, and all tensors constructed from it
        through differentiable operations.
    device: Device | "cpu" | "cuda"
        Device to store the Tensor on. Operations will be automatically performed
        on device if there is an available kernel.
    """

    def __init__(
        self,
        data: InputType,
        *,
        dtype: Dtype | None = None,
        requires_grad: bool = False,
        device: Device | DeviceLiteral = Device.Cpu,
    ):
        # Auto-detect numpy-like -- TODO: share memoryview
        if hasattr(data, "__array__") and hasattr(data, "tolist"):
            data = data.tolist()  # type: ignore

        # Strided 1D view
        device = get_std_device(device)
        dtype, shape, storage = _extract_tensor_data(data, device, dtype)
        self._dtype = dtype
        self._shape = shape
        self._storage = storage
        self._strides = _infer_strides(shape)
        self._offset = 0

        # Autograd
        if self.dtype <= Dtype.Int64 and requires_grad:
            raise RuntimeError("Cannot enable grad on a non-float tensor.")
        self._requires_grad = requires_grad
        self._grad: Tensor | None = None
        self._grad_fn: Function | None = None

    @classmethod
    def _new_contiguous(
        cls,
        storage: _C.Storage,
        shape: TensorShape,
    ) -> Tensor:
        """Intializes a new contiguous tensor from storage and shape."""
        return cls._new_view(storage, shape, None, None)

    @classmethod
    def _new_view(
        cls,
        storage: _C.Storage,
        shape: TensorShape,
        strides: TensorShape | None,
        offset: int | None,
    ) -> Tensor:
        """Intializes a new tensor view from components."""

        strides = _infer_strides(shape) if strides is None else strides
        offset = 0 if offset is None else offset
        min_index = offset + sum(
            (s - 1) * st for s, st in zip(shape, strides) if st < 0
        )
        max_index = offset + sum(
            (s - 1) * st for s, st in zip(shape, strides) if st > 0
        )
        max_storage = len(memoryview(storage))
        if math.prod(shape) > 0 and (min_index < 0 or max_index >= max_storage):
            raise IndexError(
                f"View max index {max_index} exceeds storage capacity {max_storage}."
            )

        new = cls.__new__(cls)
        new._storage = storage
        new._shape = shape
        new._strides = strides
        new._offset = offset
        new._requires_grad = False
        new._grad_fn = None
        new._grad = None
        return new

    def __repr__(self) -> str:
        if self.ndim == 0:
            return f"nt.Tensor({self.tolist()})"
        elif self.ndim == 1:
            return f"nt.Tensor({_str_list_compact(self.tolist())})"
        elif self.ndim == 2:
            if len(self) <= 5:
                indices = list(range(len(self)))
            else:
                indices = [0, 1, None, -2, -1]
            buff = "nt.Tensor("
            for ni, i in enumerate(indices):
                if ni > 0:
                    buff += "          "
                if i is None:
                    buff += "..."
                else:
                    buff += _str_list_compact(self[i].tolist()) + ""
                if ni < len(indices) - 1:
                    buff += "\n"
            return buff + ")"

        return f"nt.Tensor(shape={self.shape}, {self.numel} x <{self.dtype.name}>)"

    def __len__(self):
        """Number of elements over the first tensor dimension (row-first)."""
        if not len(self._shape):
            return 0
        return self._shape[0]

    def __getitem__(self, index: TensorIndex | tuple[TensorIndex, ...]) -> Tensor:
        if not isinstance(index, tuple):
            index = (index,)
        if self.ndim == 0:
            raise IndexError("Cannot index a scalar value.")
        if len([x for x in index if x is not None]) > self.ndim:
            raise IndexError(f"Too many indices ({len(index)}) for {self.ndim}D array.")

        index = expand_ellipsis(index, self.shape)
        view, new_index = _newview_indexing(self, index)

        if is_contiguous_view(index):
            # New view, same storage
            return Tensor._new_view(
                storage=self.storage,
                shape=view.shape,
                strides=view._strides,
                offset=view._offset,
            )

        # Sequence/masked axes necessitate mem copy
        fax = _FancyAxes.compute(view.shape, new_index)
        new_data = _C.gather_from_axes(
            x=view._C_view,
            new_sh=fax.new_shape,
            fancy_dims_in_src=fax.fancy_dims_in_src,
            fancy_dims_data=[t._C_view for t in fax.fancy_dims_data],
            out_axis_is_fancy=fax.out_axis_is_fancy,
            out_axis_target=fax.out_axis_target,
        )
        return Tensor._new_contiguous(storage=new_data, shape=fax.new_shape)

    def __setitem__(
        self, index: TensorIndex | tuple[TensorIndex, ...], value: Tensor | InputType
    ) -> None:
        if not isinstance(index, tuple):
            index = (index,)
        if self.ndim == 0:
            raise IndexError("Cannot index into a scalar tensor.")
        if len([x for x in index if x is not None]) > self.ndim:
            raise IndexError(f"Too many indices ({len(index)}) for {self.ndim}D array.")

        index = expand_ellipsis(index, self.shape)
        view, new_index = _newview_indexing(self, index)

        if not isinstance(value, Tensor):
            value = Tensor(value, dtype=self.dtype)
        elif value.dtype != self.dtype:
            value = value.to(self.dtype)

        if is_contiguous_view(index):
            try:
                value = value.expand(view.shape)
            except ValueError as e:
                raise IndexError(str(e)) from e
            if value._storage is self.storage:  # aliasing guard
                value = value._to_contiguous()
            _C.copy_view(
                src=value._C_view,
                dst=_C.TensorView(
                    self.storage, view.shape, view._strides, view._offset
                ),
            )
            return

        fax = _FancyAxes.compute(view.shape, new_index)
        try:
            value = value.expand(fax.new_shape)
        except ValueError as e:
            raise IndexError(str(e)) from e
        if value._storage is self.storage:  # aliasing guard
            value = value._to_contiguous()

        _C.scatter_to_axes(
            src=value._C_view,
            dst=self._C_view,
            fancy_dims_in_src=fax.fancy_dims_in_src,
            fancy_dims_data=[t._C_view for t in fax.fancy_dims_data],
            out_axis_is_fancy=fax.out_axis_is_fancy,
            out_axis_target=fax.out_axis_target,
        )

    @property
    def dtype(self) -> Dtype:
        """Tensor internal data type."""
        return self.storage.dtype

    @property
    def shape(self) -> TensorShape:
        """Tensor shape as an integer tuple (row-first)."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Returns the number of tensor dimensions."""
        return len(self._shape)

    @property
    def numel(self) -> int:
        """Returns the number of values in the tensor."""
        return math.prod(self.shape)

    @property
    def is_empty(self) -> bool:
        """Detects empty tensor."""
        return self.numel == 0

    @property
    def storage(self) -> _C.Storage:
        """Returns the 1D C++ storage buffer."""
        return self._storage

    @property
    def strides(self) -> tuple[TensorShape, int]:
        """Returns (strides, offset)."""
        return self._strides, self._offset

    @property
    def device(self) -> Device:
        """Return storage's device."""
        return self.storage.device

    @property
    def T(self) -> Tensor:
        """Returns a transposed view (no copy)."""
        # workaround for runtime binding of @property
        from .autograd import TransposeOp

        return TransposeOp.apply(self)

    def to(self, target: Dtype | Device | DeviceLiteral) -> Tensor:
        """Cast the tensor to a new Dtype, leaves inplace if no cast needed.

        Parameters
        ----------
        target: Dtype
            Nanotorch data type to cast the tensor to.
        """
        this = self
        if not this._is_contiguous(full_span=True):
            this = this._to_contiguous(force=True)
        if isinstance(target, Dtype):
            device = self.device
            if target == self.dtype:
                return self
            new_buffer = _C.cast(this._storage, target)
        else:
            device = get_std_device(target)
            if device == self.device:
                return self
            new_buffer = _C.to(this._storage, device)
        return Tensor._new_view(new_buffer, this.shape, this._strides, this._offset)

    def cpu(self) -> Tensor:
        """Shortcut to put tensor back on host memory."""
        return self.to(Device.Cpu)

    def clone(self) -> Tensor:
        """Returns a new copy of the tensor.

        Parameters
        ----------
        x: Tensor
            Tensor to copy.
        """
        if self._is_contiguous(full_span=True):
            return Tensor._new_contiguous(self.storage.clone(), self.shape)
        return self._to_contiguous(force=True)

    def reshape(self, *dims: int) -> Tensor:
        """Returns a reshaped view of the tensor (does not copy).

        Parameters
        ----------
        x: Tensor
            Tensor to reshape.
        dims: *int
            New shape dimensions, prod(dims) should equal tensor's number of
            elements, otherwise reshape raises a ValueError.

        Returns
        -------
        Tensor
            Reshaped tensor view.
        """
        if math.prod(dims) != self.numel:
            raise ValueError(f"Can't reshape {self.shape} into {dims}.")
        if len(dims) == 0:
            raise ValueError("No dimensions provided.")
        if dims == self.shape:
            return self
        this = self._to_contiguous()
        return Tensor._new_view(
            this.storage, shape=tuple(dims), strides=None, offset=this._offset
        )

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """Permutes two tensor dimensions (does not copy).

        Parameters
        ----------
        x: Tensor
            Tensor to transpose.
        dim0: int
            Index of the first dimension to permute.
        dim1:
            Index of the second dimension to permute.

        Returns
        -------
            Tensor view with swapped dimensions.
        """
        if len(self.shape) < 2:
            return self
        if dim0 == dim1:
            return self
        shape, strides = list(self.shape), list(self._strides)
        shape[dim0], shape[dim1], strides[dim0], strides[dim1] = (
            shape[dim1],
            shape[dim0],
            strides[dim1],
            strides[dim0],
        )
        return Tensor._new_view(
            self.storage,
            shape=tuple(shape),
            strides=tuple(strides),
            offset=self._offset,
        )

    def expand(self, shape: TensorShape) -> Tensor:
        """Expand the tensor to target shape, no copy.

        Parameters
        ----------
        shape: tuple[int, ...]
            Shape to expand the tensor to, last dimensions must match the tensor
            dimension. If tensor dimension self.shape[i] is 1, it is broadcasted to
            shape[i].
        """
        if len(shape) < self.ndim:
            raise ValueError(f"Cannot broadcast {self.shape} to {shape}.")
        strides = list(self._strides)
        for i in range(len(shape)):
            if i >= self.ndim:
                strides = [0] + strides
            else:
                src, tgt = self.shape[-i - 1], shape[-i - 1]
                if src == tgt:
                    continue
                elif src == 1:
                    strides[-i - 1] = 0  # broadcast
                else:
                    raise ValueError(
                        f"Cannot broadcast {self.shape} to {shape} (mismatch)."
                    )
        return Tensor._new_view(self.storage, shape, tuple(strides), self._offset)

    def flatten(self) -> Tensor:
        """Flattens into a 1D tensor, (2, 2, 2) -> (8,).

        Parameters
        ----------
        x: Tensor
            Tensor to flatten, shape (s1, ..., sk).

        Returns
        -------
        Tensor
            Flattened tensor view of shape (s1 * ... * sk,)
        """
        return self.reshape(self.numel)

    def equals(self, x2: Tensor) -> bool:
        """Test per-coefficient equality, auto-promotes to best dtype.

        Parameters
        ----------
        x1: Tensor
            Target tensor.
        x2: Tensor
            Target tensor.

        Returns
        -------
        bool
            Return True if tensors are equal in shape and in values (dtype can
            differ, True == 1 == 1.0).
        """
        if self.shape != x2.shape:
            return False
        if self.is_empty and x2.is_empty:
            return True
        promote_type = promote_dtypes(self.dtype, x2.dtype)
        first = self
        if promote_type != self.dtype:
            first = self.to(promote_type)
        if promote_type != x2.dtype:
            x2 = x2.to(promote_type)
        return _C.equals(first._C_view, x2._C_view)

    def tolist(self) -> InputType:
        """Converts the tensor to a scalar or list, preserving shape and type."""
        if self.device != Device.Cpu:
            raise RuntimeError("Tensor on device, call .cpu() first.")
        storage = memoryview(self.storage)

        def rec(
            shape: TensorShape, strides: TensorShape, offset: int, depth: int
        ) -> InputType:
            if depth == len(shape):
                if is_bool(self.dtype):
                    return bool(storage[offset])
                return storage[offset]
            return [
                rec(shape, strides, offset + strides[depth] * i, depth + 1)
                for i in range(shape[depth])
            ]

        return rec(self.shape, self._strides, self._offset, 0)

    def item(self) -> bool | int | float:
        """Extracts a single element tensor to a python scalar (!!expensive!!)."""
        if self.numel != 1:
            raise RuntimeError(f"Cannot convert {self} to a scalar.")
        if self.device != Device.Cpu:
            raise RuntimeError("Tensor on device, call .cpu() first.")
        storage = memoryview(self.storage)
        return storage[self._offset]

    def squeeze(self, *axes: int) -> Tensor:
        """Squeezes the specified dimensions."""
        if len(axes) == 0:
            axes = tuple(range(self.ndim))
        axes = tuple(ax if ax >= 0 else ax + self.ndim for ax in axes)
        new_shape = []
        for dim_i in range(self.ndim):
            if not (dim_i in axes and self.shape[dim_i] == 1):
                new_shape.append(self.shape[dim_i])
        return self.reshape(*new_shape)

    # Tensor ops -- bound in nanotorch.__init__

    def __add__(self, other: TensorLike) -> Tensor: ...
    def __radd__(self, other: TensorLike) -> Tensor: ...
    def __iadd__(self, other: TensorLike) -> Tensor: ...
    def __sub__(self, other: TensorLike) -> Tensor: ...
    def __rsub__(self, other: TensorLike) -> Tensor: ...
    def __isub__(self, other: TensorLike) -> Tensor: ...
    def __mul__(self, other: TensorLike) -> Tensor: ...
    def __rmul__(self, other: TensorLike) -> Tensor: ...
    def __imul__(self, other: TensorLike) -> Tensor: ...
    def __truediv__(self, other: TensorLike) -> Tensor: ...
    def __rtruediv__(self, other: TensorLike) -> Tensor: ...
    def __itruediv__(self, other: TensorLike) -> Tensor: ...
    def __neg__(self) -> Tensor: ...
    def __matmul__(self, other: Tensor) -> Tensor: ...
    def __pow__(self, exponent: TensorLike) -> Tensor: ...
    def __rpow__(self, exponent: TensorLike) -> Tensor: ...
    def __eq__(self, other: TensorLike) -> Tensor: ...
    def __gt__(self, other: TensorLike) -> Tensor: ...
    def __ge__(self, other: TensorLike) -> Tensor: ...
    def __lt__(self, other: TensorLike) -> Tensor: ...
    def __le__(self, other: TensorLike) -> Tensor: ...
    def exp(self) -> Tensor: ...
    def log(self) -> Tensor: ...
    def pow(self, exponent: TensorLike) -> Tensor: ...
    def sum(
        self,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        *,
        dtype: Dtype | None = None,
    ) -> Tensor: ...
    def mean(
        self,
        axis: int | TensorShape | None = None,
        keepdim: bool = False,
        *,
        dtype: Dtype | None = None,
    ) -> Tensor: ...

    # Autograd

    @property
    def requires_grad(self) -> bool:
        """This tensor tracks gradients."""
        return self._requires_grad or self.grad_fn is not None

    @property
    def is_leaf(self) -> bool:
        """This tensor does not come from other tensors."""
        return self.grad_fn is None

    @property
    def grad(self) -> Tensor | None:
        """Stored gradients, if any."""
        return self._grad

    @property
    def grad_fn(self) -> Function | None:
        """If non-leaf tensor, grad-bearing function to backprop through."""
        return self._grad_fn

    def backward(self) -> None:
        """Backpropagate scalar tensor gradients through the gradient graph."""
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call .backward() on a tensor that does not require grad "
                "and does not have a grad_fn."
            )
        if self.numel != 1:
            raise RuntimeError(
                f"Cannot call .backward() on a non-scalar tensor of shape {self.shape}"
            )
        self._grad = Tensor(1.0, dtype=self.dtype)
        if self.shape != ():
            self._grad = self._grad.reshape(*self.shape)
        if not self.is_leaf:
            _backpropagate_grad(self)

    def attach_grad_fn(self, grad_fn: Function) -> None:
        """Attach a differentiable functions and parent tensors."""
        self._grad_fn = grad_fn

    def zero_grad(self) -> None:
        """Reset grad to 0."""
        self._grad = None

    def enable_grad(self) -> None:
        """Enables gradient tracking post-constructor."""
        self._requires_grad = True

    # Private operators

    def _is_contiguous(self, full_span: bool = False) -> bool:
        """Checks if current view is contiguous in memory."""
        if not full_span:
            return self._strides == _infer_strides(self._shape)
        return (
            self._strides == _infer_strides(self._shape)
            and self._offset == 0
            and self.numel == len(memoryview(self.storage))
        )

    def _to_contiguous(self, force: bool = False) -> Tensor:
        """Returns a contiguous version of the tensor (copy if necessary)."""
        if not force and self._is_contiguous():
            return self
        return Tensor(self.tolist(), dtype=self.dtype)

    @property
    def _C_view(self) -> _C.TensorView:
        """Return a C++-compatible tensor view."""
        return _C.TensorView(self.storage, self.shape, self._strides, self._offset)

    def _alias(self) -> Tensor:
        """Returns a new Tensor wrapper (no copy)."""
        return Tensor._new_view(self.storage, self.shape, *self.strides)


# Private functions


def _extract_tensor_data(
    data: InputType,
    target_device: Device,
    user_dtype: Dtype | None = None,
) -> tuple[Dtype, TensorShape, _C.Storage]:
    """Automatically extract tensor information."""

    # Accumulators
    shape: list[int] = []
    flat: list[float | int | bool] = []
    auto_dtype: Dtype | None = None
    leaf_level: int | None = None

    def rec(data: InputType, level: int) -> None:
        # Recursive walk through the data
        nonlocal auto_dtype
        nonlocal leaf_level

        if level > MAX_TENSOR_LEVEL:
            raise ValueError(f"Maximum tensor depth reached: {MAX_TENSOR_LEVEL}")
        if isinstance(data, (bool, int, float)):
            if leaf_level is None:
                leaf_level = level
            elif level != leaf_level:
                raise ValueError("Leaf found at non-terminal level.")
            dtype = dtype_from_type(data)
            if auto_dtype is None or dtype > auto_dtype:
                auto_dtype = dtype
            flat.append(data)
        elif isinstance(data, list):
            nnodes = len(data)
            if level >= len(shape):
                shape.append(nnodes)
            elif shape[level] != nnodes:
                raise ValueError(
                    f"Unhomogeneous tensor shape detected ({shape[level]} != {nnodes})"
                )
            for node in data:
                rec(node, level + 1)
        else:
            raise TypeError(f"Unsupported tensor element type: {type(data)}")

    rec(data, 0)

    # Final type casting
    auto_dtype = auto_dtype or Dtype.Float32
    if user_dtype is not None:
        final_dtype = user_dtype
    else:
        final_dtype = auto_dtype

    # Needs int cast
    if is_bool(final_dtype):
        values = list(bool(x) for x in flat)
    elif is_float(auto_dtype) and is_int(final_dtype):
        values = list(int(x) for x in flat)
    else:
        values = flat

    return (
        final_dtype,
        tuple(shape),
        _C.Storage.from_iterable(values, final_dtype, target_device),
    )


def _infer_strides(shape: TensorShape) -> TensorShape:
    """Infer major strides from shape."""
    strides = []
    for i in range(len(shape)):
        if i == 0:
            new_stride = 1
        else:
            new_stride = strides[0] * shape[-i]
        strides.insert(0, new_stride)
    return tuple(strides)


def _str_list_compact(li: Any) -> str:
    """Returns a compact list string."""
    if not isinstance(li, list):
        return str(li)
    if len(li) < 8:
        return str(li)
    return (
        "["
        + ", ".join(str(x) for x in li[:3])
        + ", ..., "
        + ", ".join(str(x) for x in li[-3:])
        + "]"
    )


@dataclass
class _FancyAxes:
    """Container for get_fancy_axes helper."""

    new_shape: TensorShape
    fancy_dims_in_src: list[int]
    fancy_dims_data: list[Tensor]
    out_axis_is_fancy: list[bool]
    out_axis_target: list[int]

    @classmethod
    def compute(cls, shape: TensorShape, index: tuple[TensorIndex, ...]) -> _FancyAxes:
        fancy_dims_in_src: list[int] = []
        basic_axes: list[int] = []
        basic_shape: list[int] = []
        indarrs_raw: list[Tensor] = []

        for dim in range(len(shape)):
            # Non-fancy axis, skip
            if (
                dim >= len(index)
                or index[dim] is None
                or isinstance(index[dim], (int, slice))
            ):
                basic_axes.append(dim)
                basic_shape.append(shape[dim])
                continue

            # Fancy axis (integer-based or boolean mask)
            fancy_index = index[dim]
            if isinstance(fancy_index, Tensor):
                pass
            elif isinstance(fancy_index, Sequence):
                try:
                    fancy_index = Tensor(fancy_index)  # type: ignore
                except Exception as e:
                    raise IndexError(
                        f"Could not interpret index {dim} as a Tensor: {e}"
                    )
            else:
                raise IndexError(f"Unexpected selector type {type(fancy_index)}")

            # TODO: do it index-based once implemented
            if not fancy_index._is_contiguous(full_span=True):
                fancy_index = fancy_index._to_contiguous(force=True)
            content = memoryview(fancy_index._storage)
            for i in range(len(content)):
                if content[i] < 0:
                    content[i] = content[i] + shape[dim]
                if content[i] < 0 or content[i] >= shape[dim]:
                    raise IndexError(
                        f"Invalid index {content[i]} in axis of len {shape[dim]}."
                    )

            if fancy_index.dtype == Dtype.Bool:
                if fancy_index.ndim > 1:
                    raise NotImplementedError("nD boolean masks not here (yet).")
                if len(fancy_index) != shape[dim]:
                    raise IndexError(
                        f"Wrong boolean index shape: {len(fancy_index)} (expected {shape[dim]})."
                    )
                indices = [i for i, b in enumerate(content) if b]
                fancy_index = Tensor(indices)

            fancy_dims_in_src.append(dim)
            indarrs_raw.append(fancy_index.to(Dtype.Int64))

        indarr_shape = broadcast_shapes(*(t.shape for t in indarrs_raw))
        fancy_dims_data = [t.expand(indarr_shape) for t in indarrs_raw]

        # compute new shape and axes plan
        nfancy = len(fancy_dims_in_src)
        nidx = len(indarr_shape)
        axes_diff = [
            fancy_dims_in_src[i + 1] - fancy_dims_in_src[i] for i in range(nfancy - 1)
        ]

        out_axis_is_fancy = []
        out_axis_target = []
        if nfancy > 0 and any(d != 1 for d in axes_diff):  # Not contiguous case
            new_shape = tuple(list(indarr_shape) + basic_shape)
            for i in range(len(new_shape)):
                if i < nidx:
                    out_axis_is_fancy.append(True)
                    out_axis_target.append(i)
                else:
                    out_axis_is_fancy.append(False)
                    out_axis_target.append(basic_axes[i - nidx])
        else:
            a0 = fancy_dims_in_src[0]
            new_shape = tuple(basic_shape[:a0] + list(indarr_shape) + basic_shape[a0:])
            for i in range(len(new_shape)):
                if i < a0:
                    out_axis_is_fancy.append(False)
                    out_axis_target.append(basic_axes[i])
                elif i < a0 + nidx:
                    out_axis_is_fancy.append(True)
                    out_axis_target.append(i - a0)
                else:
                    out_axis_is_fancy.append(False)
                    out_axis_target.append(basic_axes[i - nidx])

        return _FancyAxes(
            new_shape=new_shape,
            fancy_dims_in_src=fancy_dims_in_src,
            fancy_dims_data=fancy_dims_data,
            out_axis_is_fancy=out_axis_is_fancy,
            out_axis_target=out_axis_target,
        )


def _newview_indexing(
    x: Tensor, index: tuple[TensorIndex, ...]
) -> tuple[Tensor, tuple[TensorIndex, ...]]:
    """Indexes a new view of the same storage (simple indexing)."""
    offset = x._offset
    new_shape: list[int] = []
    new_strides: list[int] = []
    new_index: list[TensorIndex] = []

    olddim = 0
    newdim = 0
    while olddim < len(x.shape) or newdim < len(index):
        if newdim >= len(index) or (
            (not isinstance(index[newdim], (int, slice)))
            and (index[newdim] is not None)
        ):
            new_shape.append(x.shape[olddim])
            new_strides.append(x._strides[olddim])
            new_index.append(None if newdim >= len(index) else index[newdim])
            olddim += 1
            newdim += 1
            continue

        sel = index[newdim]
        if sel is None:
            new_shape.append(1)
            new_strides.append(0)
            new_index.append(None)
            newdim += 1
        else:
            nelem = x.shape[olddim]
            if isinstance(sel, int):
                if sel < 0:
                    sel += nelem
                if sel < 0 or sel >= nelem:
                    raise IndexError(
                        f"Cannot index at position {sel} in {nelem} elements."
                    )
                offset += sel * x._strides[olddim]
            elif isinstance(sel, slice):
                start, stop, step = sel.indices(nelem)
                length = len(range(start, stop, step))
                new_shape.append(length)
                new_strides.append(x._strides[olddim] * step)
                new_index.append(None)
                offset += start * x._strides[olddim]
            else:
                raise ValueError(f"Unhandle index type: {type(sel)}.")
            olddim += 1
            newdim += 1

    return Tensor._new_view(
        x.storage, tuple(new_shape), tuple(new_strides), offset
    ), tuple(new_index)


# Backprop


def _backpropagate_grad(seed: Tensor) -> None:
    """Backpropagate a value through differentiable functions."""
    # Gather nodes via BFS
    to_visit = [seed]
    ncons = {id(seed): 0}
    sorted_nodes: list[tuple[Tensor, Function, tuple[Tensor, ...]]] = []
    while to_visit:
        f_x = to_visit.pop(0)
        grad_fn = f_x.grad_fn
        if grad_fn is None:
            continue
        inputs = []
        for x in grad_fn.inputs:
            if isinstance(x, Tensor):
                if x.requires_grad:
                    ncons.setdefault(id(x), 0)
                    ncons[id(x)] += 1
                    to_visit.append(x)
                inputs.append(x)
        sorted_nodes.append((f_x, grad_fn, tuple(inputs)))

    # Propagate the gradients
    for f_x, grad_fn, inputs in sorted_nodes:
        if f_x.grad is None:
            raise RuntimeError("f_x.grad is None.")
        ncons[id(f_x)] -= 1
        if ncons[id(f_x)] > 0:
            continue
        grads = grad_fn.backward(f_x.grad)
        for x, grad in zip(inputs, grads):
            if not x.requires_grad:
                continue
            if x._grad is None:
                x._grad = grad
            else:
                x._grad += grad
        f_x._grad = None


TensorIndex = (
    int | slice | Sequence[bool | int | Sequence[int]] | EllipsisType | Tensor | None
)
TensorLike = Tensor | float | int | bool | list


def inherit_doc(source):
    """Copies the docstring of source into the decorated element.

    Usage
    -----
    >>> def f1():
    >>>     '''Some docstring'''
    >>>     return 0

    >>> @inherit_doc(f1)
    >>> def f2():
    >>>     return 1

    >>> print(f2.__doc__)
    >>> '''Some docstring'''
    """

    def wrap(fn):
        fn.__doc__ = source.__doc__
        return fn

    return wrap
