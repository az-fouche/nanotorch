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
addop             203.49M        1.52G         7.5x        0.00%
expop             587.21M       89.10G       151.7x        0.16%
logop             505.73M       92.41G       182.7x        0.17%
matmul-torch      854.66G       38.08T        44.6x       68.00%
matmulop            2.46G       19.30T      7830.8x       34.46%
maxop             202.00M      115.16G       570.1x        0.21%
meanop            212.33M       71.48G       336.6x        0.13%
minop             202.87M      113.43G       559.1x        0.20%
mulop             203.05M        1.52G         7.5x        0.00%
negop              10.29G       97.39G         9.5x        0.17%
powop             204.04M       86.92G       426.0x        0.16%
reluop             10.92G       96.27G         8.8x        0.17%
sigmoidop          88.38M       91.94G      1040.2x        0.16%
sqrtop              5.97G       90.86G        15.2x        0.16%
subop             206.92M        1.52G         7.4x        0.00%
sumop             223.03M      121.17G       543.3x        0.22%
tanhop            499.05M       95.30G       191.0x        0.17%
truedivop         206.20M        1.52G         7.4x        0.00%
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