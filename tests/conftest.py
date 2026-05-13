import pytest

import nanotorch as nt

requires_cuda = pytest.mark.skipif(not nt.is_available(), reason="CUDA not available")


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not nt.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return request.param
