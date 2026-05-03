import time

import nanotorch as nt

N_REP = 100


def main():
    x = nt.rand(10_000, 512, 10)

    print("== CPU benchmark ==")
    t0 = time.time()
    for _ in range(100):
        x.exp()
    duration = time.time() - t0
    print(f"CPU time: {duration:.2f}s ({N_REP / duration:.2f} ops/s)")

    print("== GPU benchmark ==")
    tw = time.time()
    x = x.to("cuda")
    x.exp()
    t0 = time.time()
    for _ in range(100):
        y = x.exp()
    y.cpu()  # forces sync
    duration_w = time.time() - tw
    duration = time.time() - t0
    print(f"GPU time (+warmup): {duration_w:.2f}s ({N_REP / duration_w:.2f} ops/s)")
    print(f"GPU time (pure kernel): {duration:.2f}s ({N_REP / duration:.2f} ops/s)")


if __name__ == "__main__":
    main()
