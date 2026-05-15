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
addop             204.28M        1.39G         6.8x        0.00%
expop             559.28M       11.69G        20.9x        0.02%
logop             496.38M       12.00G        24.2x        0.02%
matmul-torch      911.90G       38.08T        41.8x       68.00%
matmulop            2.51G       16.53T      6589.3x       29.51%
maxop             208.99M       46.00G       220.1x        0.08%
meanop            220.45M       34.56G       156.8x        0.06%
minop             208.67M       48.10G       230.5x        0.09%
mulop             213.13M        1.42G         6.6x        0.00%
negop               8.34G       12.18G         1.5x        0.02%
powop             215.21M       11.86G        55.1x        0.02%
reluop              8.36G       12.19G         1.5x        0.02%
sigmoidop          88.81M       11.08G       124.8x        0.02%
sqrtop              3.70G       11.68G         3.2x        0.02%
subop             212.63M        1.41G         6.6x        0.00%
sumop             224.24M       47.72G       212.8x        0.09%
tanhop            475.07M       13.74G        28.9x        0.02%
truedivop         205.16M        1.41G         6.9x        0.00%
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