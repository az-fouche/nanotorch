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

I also added a relative comparison to the theoretical peak FLOPS and memory bandwidth on this device (`fp32`). As expected, raw kernel launch without fusion is severely memory-bound! But I'll update this table as I optimize each kernel, to see where it lands. Keep in mind that libraries like cuBLAS endow dozens of kernel versions for each operation, with advanced heuristics to choose the most efficient one in every situation!

```
op            cpu (FLOPS) cuda (FLOPS)   %peak(flops)   cuda (mem)   %peak(mem)
-------------------------------------------------------------------------------
addop               1.36G       68.04G         0.12%      816.53G        91.13%
expop             470.83M       99.45G         0.18%      795.56G        88.79%
logop             434.14M       79.23G         0.14%      633.85G        70.74%
matmul-torch      887.00G       37.74T        67.39%       37.74G         4.21%
matmulop            2.43G       16.54T        29.53%       24.80G         2.77%
maxop             261.42M       97.11G         0.17%      388.44G        43.35%
meanop            272.21M       95.80G         0.17%      383.21G        42.77%
minop             271.12M      103.42G         0.18%      413.66G        46.17%
mulop               1.27G       64.85G         0.12%      778.19G        86.85%
negop               1.27G       79.08G         0.14%      632.62G        70.60%
powop             168.53M       80.79G         0.14%      646.32G        72.13%
reluop              1.26G       73.68G         0.13%      589.46G        65.79%
sigmoidop          83.08M       79.00G         0.14%      632.02G        70.54%
sqrtop              1.34G       66.24G         0.12%      529.95G        59.15%
subop               1.29G       52.91G         0.09%      634.90G        70.86%
sumop             273.56M       80.66G         0.14%      322.66G        36.01%
tanhop            416.44M       83.82G         0.15%      670.57G        74.84%
truedivop           1.32G       53.71G         0.10%      644.51G        71.93%
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
