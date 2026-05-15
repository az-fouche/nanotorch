# Nanotorch: a minimal PyTorch clone

My pet project of the moment. The goal here is to reimplement from scratch a PyTorch-like imperative autograd library with all the expected primitives and CUDA support.

## Nanotorch scope

- Working autograd engine.
- Tensor storage handled in contiguous memory with strided views.
- Intensive operations bound from C++ implementations.
- Partial CUDA support with acceptable performance (we don't aim for 90% of cuBLAS performance though).
- Deep learning boilerplate (modules, optimizers, dataloaders...).
- Comprehensive test coverage.

## Setup

```bash
# Install the environment
uv sync

# Launch a MLP training on mock data (but real compute)
uv run python scripts/train_mlp_basic.py --device [cpu|cuda]

# Launch the unit tests
uv run pytest 
```

## Current operations throughput

Launch the profiling tool with the following command:

```bash
$ python scripts\profile_ops.py
```

It runs a pass on all the autograd operations both on CPU and GPU and measures the FLOPS throughput for each kernel -- note that it includes some python overhead as well so that's not 1:1 comparable to pure C++ profiling. Here is the current state of the library, measured with a consumer-grade GPU (RTX 5080). 

For fun, I also added a comparison to the theoretical peak FLOPS on this device (`fp32`). As expected, raw kernel launch without fusion is severely memory-bound! But I'll update this table as I optimize each kernel, to see where it lands (and to underline how strong cuBLAS engineers are). 

```
op            cpu (FLOPS) cuda (FLOPS)      speedup        %peak
----------------------------------------------------------------
addop             208.23M        1.40G         6.7x        0.00%
expop             605.07M       12.28G        20.3x        0.02%
logop             520.41M       11.58G        22.3x        0.02%
matmul-torch      963.45G       38.19T        39.6x       68.19%
matmulop            2.45G       19.66T      8036.6x       35.10%
maxop             205.69M       47.51G       231.0x        0.08%
meanop            214.39M       35.27G       164.5x        0.06%
minop             207.83M       46.39G       223.2x        0.08%
mulop             213.77M        1.40G         6.5x        0.00%
negop              11.34G       14.63G         1.3x        0.03%
powop             210.94M       14.48G        68.7x        0.03%
reluop             10.52G       14.85G         1.4x        0.03%
sigmoidop          91.05M       11.53G       126.7x        0.02%
sqrtop              5.63G       14.82G         2.6x        0.03%
subop             213.29M        1.33G         6.2x        0.00%
sumop             226.50M       46.41G       204.9x        0.08%
tanhop            506.21M       14.84G        29.3x        0.03%
truedivop         217.93M        1.43G         6.5x        0.00%
```

## Generative AI stance

**No generative AI has been used to produce the code** (this is a hard project constraint). I believe that while coding agents are great tools, they can also limit the learning process in such project. 

For this reason, I use Claude Code here as a "pair programming" companion: it helps me debug, reviews my code, explores the docs, but it does not take the technical decisions nor write the software. This is (for now) the best balance I found to maximize the personal benefit of these systems. 

## References

- [*PyTorch internals*, E. Zyang's blog post](https://blog.ezyang.com/2019/05/pytorch-internals/) -- A good starting point to explore PyTorch's architecture
- [*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*, S. Boehm's blog post](https://siboehm.com/articles/22/CUDA-MMM) -- How to incrementally approach cuBLAS' reference performance of GEMM
- [*Optimizing Parallel Reduction in CUDA*, Mark Harris](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) -- How to implement fast parallel reduction algorithms 
- [TinyGrad](https://tinygrad.org/#tinygrad) -- A really thought-provoking framework that proposes to rethink the way we approach device programming
- [*CUDA C++ Programming Guide*, NVIDIA technical document](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf) -- The CUDA Bible, literally
- [PyTorch's Github](https://github.com/pytorch/pytorch) -- Though the repo has become quite gigantic and it's not easy to find what I look for :)