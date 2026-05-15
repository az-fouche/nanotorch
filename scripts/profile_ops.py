"""Profile the speed and FLOPS of all package operations."""

import argparse
import random
import re
import time
from dataclasses import dataclass

import torch
import tqdm

import nanotorch as nt
from nanotorch import Tensor
from nanotorch import autograd as ag
from nanotorch.autograd.ops_spec import gen_random_input_for

N_CALLS = 100

RTX_5080_FP32_PEAK = 56e12  # TFLOPS

CUDA_AVAILABLE = nt.cuda.is_available()


@dataclass(kw_only=True)
class ProfilingResults:
    """Profiling results of a single spec."""

    op_name: str
    cpu_flops: float
    cuda_flops: float


def profile_op_cpu(op: type[ag.Function]) -> float | None:
    """Profile the flops of a single op on CPU."""
    nt.manual_seed(42)
    random.seed(42)
    inputs = gen_random_input_for(
        op.op_spec, min_ndim=2, max_ndim=2, min_size=500, max_size=500
    )
    total_t = 0
    flops = op.flops(*inputs)
    if flops == 0:
        return None
    for i in range(N_CALLS):
        t0 = time.perf_counter()
        op.apply(*inputs)
        if i < 2:  # Skip warmup
            continue
        total_t += time.perf_counter() - t0
    return flops * (N_CALLS - 2) / total_t


def profile_op_cuda(op: type[ag.Function]) -> float:
    """Profile the flops of a single op on CUDA."""
    nt.manual_seed(42)
    random.seed(42)
    inputs = [
        x.to("cuda") if isinstance(x, Tensor) else x
        for x in gen_random_input_for(
            op.op_spec, min_ndim=2, max_ndim=2, min_size=4000, max_size=4000
        )
    ]
    flops = op.flops(*inputs)
    if flops == 0:
        return 0.0
    total_t = 0
    for i in range(N_CALLS):
        t0 = time.perf_counter()
        op.apply(*inputs)
        if i < 2:  # Skip warmup
            continue
        nt.cuda.sync()
        total_t += time.perf_counter() - t0
    return flops * (N_CALLS - 2) / total_t


def profile_torch_matmul() -> ProfilingResults:
    """Profile the wall time of a single op."""
    nt.manual_seed(42)
    random.seed(42)
    A_cpu, B_cpu, A_cuda, B_cuda = (
        torch.rand(500, 500),
        torch.rand(500, 500),
        torch.rand(6000, 6000).to("cuda"),
        torch.rand(6000, 6000).to("cuda"),
    )
    flops_cpu = ag.MatmulOp.flops(A_cpu, B_cpu)  # type: ignore
    flops_cuda = ag.MatmulOp.flops(A_cuda, B_cuda)  # type: ignore

    total_t_cpu, total_t_cuda = 0, 0
    for i in range(N_CALLS):
        t0 = time.perf_counter()
        _ = A_cpu @ B_cpu
        if i >= 2:
            total_t_cpu += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = A_cuda @ B_cuda
        torch.cuda.synchronize()
        if i >= 2:
            total_t_cuda += time.perf_counter() - t0

    return ProfilingResults(
        op_name="matmul-torch",
        cpu_flops=flops_cpu * (N_CALLS - 2) / total_t_cpu,
        cuda_flops=flops_cuda * (N_CALLS - 2) / total_t_cuda,
    )


def should_run(op_name: str, filter: str | None, regex: bool):
    """Detect if op name passes the filter."""
    if filter is None:
        return True
    if not regex:
        return op_name.lower() == filter.lower()
    return re.search(filter, op_name) is not None


def fmt_flops(flops: float) -> str:
    """Human-readable flops."""
    if flops > 1e12:
        return f"{flops / 1e12:.2f}T"
    elif flops > 1e9:
        return f"{flops / 1e9:.2f}G"
    elif flops > 1e6:
        return f"{flops / 1e6:.2f}M"
    elif flops > 1e3:
        return f"{flops / 1e3:.2f}K"
    return f"{flops:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter ops name using this parameter.",
    )
    parser.add_argument("-E", action="store_true", help="Enable regex filtering.")
    args = parser.parse_args()
    results = []
    pbar = tqdm.tqdm(ag.ALL_OPS_, ncols=80)
    for op in pbar:
        op_name = op.__name__.lower()
        if not should_run(op_name, args.filter, args.E):
            continue
        with nt.no_grad():
            pbar.set_description(f"Profiling {op_name} (cpu)")
            cpu_flops = profile_op_cpu(op)
            if cpu_flops is None:  # Virtual ops like reshape
                continue
            if CUDA_AVAILABLE:
                pbar.set_description(f"Profiling {op_name} (cuda)")
                cuda_flops = profile_op_cuda(op)
            else:
                cuda_flops = 0.0
        results.append(
            ProfilingResults(
                op_name=op_name, cpu_flops=cpu_flops, cuda_flops=cuda_flops
            )
        )
    results.append(profile_torch_matmul())

    results = sorted(results, key=lambda r: r.op_name)

    hdr = f"{'op':<12} {'cpu (FLOPS)':>12} {'cuda (FLOPS)':>12} {'speedup':>12} {'%peak':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        sp = r.cuda_flops / r.cpu_flops if r.cpu_flops > 0 else float("inf")
        print(
            f"{r.op_name:<12} {fmt_flops(r.cpu_flops):>12} "
            f"{fmt_flops(r.cuda_flops):>12} "
            f"{sp:>11.1f}x"
            f"{r.cuda_flops / RTX_5080_FP32_PEAK * 100:>12.2f}%"
        )


if __name__ == "__main__":
    main()
