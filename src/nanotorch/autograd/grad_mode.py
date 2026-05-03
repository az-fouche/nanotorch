"""No grad/inference context."""

_grad_enabled = True


def is_grad_enabled() -> bool:
    """If False, gradients should not be tracked."""
    return _grad_enabled


class no_grad:
    def __init__(self) -> None:
        self._init_state = _grad_enabled

    def __enter__(self):
        global _grad_enabled
        _grad_enabled = False

    def __exit__(self, exc_type, exc, tb):
        global _grad_enabled
        _grad_enabled = self._init_state
