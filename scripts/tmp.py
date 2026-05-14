import time

import nanotorch as nt

x = nt.rand(4000, 4000).to("cuda")
for y in (
    nt.tensor(3.14).to("cuda"),
    nt.rand(1, 4000).to("cuda"),
    nt.rand(4000, 4000).to("cuda"),
):
    nt.cuda.sync()
    N = 20
    for _ in range(2):
        _ = x + y  # warmup
    nt.cuda.sync()
    t0 = time.perf_counter()
    for _ in range(N):
        _ = x + y
    nt.cuda.sync()
    print(
        f"{x.shape} + {y.shape}: {16e6 * N / (time.perf_counter() - t0) / 1e9:.2f}G FLOPS"
    )
