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
addop             168.31M        1.28G         7.6x        0.00%
expop             534.15M       11.04G        20.7x        0.02%
logop             482.54M       11.03G        22.9x        0.02%
matmulop            2.26G       17.96T      7956.7x       32.07%
meanop            205.36M       25.94G       126.3x        0.05%
mulop             199.55M        1.39G         7.0x        0.00%
negop               7.04G       11.67G         1.7x        0.02%
powop             196.17M       12.77G        65.1x        0.02%
reluop              6.31G       12.87G         2.0x        0.02%
sigmoidop          86.10M       12.56G       145.9x        0.02%
sqrtop              4.67G       13.07G         2.8x        0.02%
subop             202.94M        1.11G         5.4x        0.00%
sumop             197.27M       48.07G       243.7x        0.09%
tanhop            492.62M       12.90G        26.2x        0.02%
truedivop         193.41M        1.41G         7.3x        0.00%
minop             195.61M       44.03G       225.1x        0.08%
maxop             199.54M       43.92G       220.1x        0.08%
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