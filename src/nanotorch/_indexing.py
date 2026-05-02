"""Flat storage indexing machinery."""

from types import EllipsisType

TensorShape = tuple[int, ...]


def broadcast_shapes(*shapes: TensorShape) -> TensorShape:
    """Broadcast to the smallest compatible shape, raises if not possible."""
    if not shapes:
        return tuple()
    final_shape = list(shapes[0])
    for s in shapes[1:]:
        ldiff = len(final_shape) - len(s)
        if ldiff > 0:
            s = [1] * ldiff + list(s)
        elif ldiff < 0:
            final_shape = [1] * -ldiff + final_shape
        for i in range(len(final_shape)):
            if final_shape[i] == 1:
                final_shape[i] = s[i]
            elif s[i] == 1:
                continue
            elif final_shape[i] != s[i]:
                raise RuntimeError(f"Cannot broadcast shapes {s} and {final_shape}.")
    return tuple(final_shape)


def expand_ellipsis(index: tuple, shape: TensorShape) -> tuple:
    """Expands ellipsis into slice [..., 0] -> [:, :, :, 0]"""
    if len(index) == 0:
        return ()
    if sum(isinstance(i, EllipsisType) for i in index) > 1:
        raise IndexError("Multiple ellipses detected.")
    ndim = len(shape)
    nind = sum(i is not None for i in index)
    if isinstance(index[0], EllipsisType):
        return tuple([slice(None)] * (ndim - nind + 1) + list(index[1:]))
    if isinstance(index[-1], EllipsisType):
        return tuple(list(index[:-1]) + [slice(None)] * (ndim - nind + 1))
    pointer = 0
    index_exp = []
    while pointer < len(index):
        elem = index[pointer]
        if not isinstance(elem, EllipsisType):
            index_exp.append(elem)
        else:
            index_exp.extend([slice(None)] * (ndim - nind + 1))
        pointer += 1
    return tuple(index_exp)


def is_contiguous_view(index: tuple) -> bool:
    """Computes if the indexing can be done without copy."""
    for idx in index:
        if not isinstance(idx, (int, slice)) and idx is not None:
            return False
    return True
