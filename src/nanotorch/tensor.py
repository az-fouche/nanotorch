import typing

TensorType = list | float


class Tensor:
    """Base tensor class.

    Parameters
    ----------
    data: TensorType
        Container (scalar or list) of the tensor coefficients.
    """

    def __init__(self, data: TensorType):

        if not isinstance(data, TensorType):
            raise ValueError(
                f"Unexpected data type: {type(data)}. Expected a type among "
                f"{', '.join(t.__name__ for t in typing.get_args(TensorType))}"
            )
        _assert_is_tensor_shape(data)
        self._data = data

    def __repr__(self) -> str:
        """String representation."""
        return f"NtTensor({_stringify(self._data)})"

    def __len__(self) -> int:
        """Return tensor length over the first dimension."""
        shape = self.shape
        if not len(shape):
            return 0
        return shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return tensor shape over each dimension."""
        dims = []
        nested = self._data
        while isinstance(nested, list):
            dims.append(len(nested))
            if not len(nested):
                break
            nested = nested[0]
        return tuple(dims)

    @property
    def dim(self) -> int:
        """Return tensor dimension."""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """Return total number of elements."""
        nelem = 1
        for dim_i in self.shape:
            nelem *= dim_i
        return nelem


def tensor(data: TensorType) -> Tensor:
    """Create a new tensor.

    Parameters
    ----------
    data: TensorType
        Container (scalar or list) of the tensor coefficients.
    """
    return Tensor(data)


def _stringify(data: TensorType) -> str:
    """Converts a tensor into a formatted string."""
    if not isinstance(data, list):
        return str(data)
    return f"[{', '.join(_stringify(row) for row in data)}]"


def _assert_is_tensor_shape(data: TensorType) -> None:
    """Raises an exception if the data is misshaped."""
    if not isinstance(data, list):
        return

    data_type: typing.Type | None = None
    inner_size: int | None = None
    for row in data:
        # Type checking
        if not isinstance(row, TensorType):
            raise ValueError(f"Non-tensor type detected: {type(row)}")
        elif data_type is None:
            data_type = type(row)
        elif not isinstance(row, data_type):
            raise ValueError(
                f"Unhomogeneous data type: {type(row)} (expected {data_type})"
            )

        # Shape checking
        if isinstance(row, list):
            inner_size_i = len(row)
        else:
            inner_size_i = 0
        if inner_size is None:
            inner_size = inner_size_i
        elif inner_size_i != inner_size:
            raise ValueError(
                f"Unhomogeneous inner size: {inner_size_i} (expected {inner_size})"
            )

        _assert_is_tensor_shape(row)
