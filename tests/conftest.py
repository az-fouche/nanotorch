import pytest

import nanotorch as nt

requires_cuda = pytest.mark.skipif(
    not nt.is_cuda_available(), reason="CUDA not available"
)
