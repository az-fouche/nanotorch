"""Profile the speed and FLOPS of all package operations."""

import time
from dataclasses import dataclass

import tqdm

import nanotorch as nt
from nanotorch import Tensor
from nanotorch import autograd as ag
from nanotorch.autograd.ops_spec import gen_random_input_for

N_CALLS = 100

CUDA_AVAILABLE = nt.cuda.is_available()


@dataclass(kw_only=True)
class ProfilingResults:
    """Profiling results of a single spec."""

    op_name: str
    cpu_wtime: float
    cuda_wtime: float


def profile_op(op: type[ag.Function], device: str) -> float:
    """Profile the wall time of a single op."""
    total_t = 0
    for i in range(N_CALLS):
        inputs = [
            x.to(device) if isinstance(x, Tensor) else x
            for x in gen_random_input_for(op.op_spec, max_dim=3, size_factor=50)
        ]
        nt.cuda.sync()
        t0 = time.time()
        op.apply(*inputs)
        if i == 0:  # Skip warmup
            continue
        nt.cuda.sync()
        total_t += time.time() - t0
    return total_t


def main():
    results = []
    pbar = tqdm.tqdm(ag.ALL_OPS_, ncols=80)
    for op in pbar:
        op_name = op.__name__.lower()
        pbar.set_description(f"Profiling {op_name} (cpu)")
        t_cpu = profile_op(op, "cpu")
        if CUDA_AVAILABLE:
            pbar.set_description(f"Profiling {op_name} (cuda)")
            t_cuda = profile_op(op, "cuda")
        else:
            t_cuda = 0.0
        results.append(
            ProfilingResults(op_name=op_name, cpu_wtime=t_cpu, cuda_wtime=t_cuda)
        )
    hdr = f"{'op':<12} {'cpu (s)':>10} {'cuda (s)':>10} {'speedup':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        sp = r.cpu_wtime / r.cuda_wtime if r.cuda_wtime else float("inf")
        print(f"{r.op_name:<12} {r.cpu_wtime:>10.4f} {r.cuda_wtime:>10.4f} {sp:>9.1f}x")


if __name__ == "__main__":
    main()
