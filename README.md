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
addop             166.48M        1.40G         8.4x        0.00%
expop             583.77M        7.76G        13.3x        0.01%
logop             511.35M       11.59G        22.7x        0.02%
matmulop            2.51G       11.99T      4774.4x       21.41%
meanop            192.71M       34.90G       181.1x        0.06%
mulop             202.98M        1.40G         6.9x        0.00%
negop               7.87G       12.11G         1.5x        0.02%
powop             189.97M       11.48G        60.5x        0.02%
reluop              7.61G       12.11G         1.6x        0.02%
sigmoidop          85.30M       11.91G       139.6x        0.02%
sqrtop              4.87G       11.74G         2.4x        0.02%
subop             200.16M        1.42G         7.1x        0.00%
sumop             199.44M       45.33G       227.3x        0.08%
tanhop            497.61M       12.03G        24.2x        0.02%
truedivop         194.00M        1.42G         7.3x        0.00%
minop             185.79M       43.86G       236.1x        0.08%
maxop             196.06M       47.41G       241.8x        0.08%
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