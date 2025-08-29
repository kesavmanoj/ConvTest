# ConvTest Quick Start Guide

## What is ConvTest?

ConvTest is a CUDA learning project that demonstrates 2D convolution optimization techniques on GPU hardware. It's organized as a progressive learning path from basic GPU setup to advanced optimization techniques.

## Quick Start

### 1. Check Your Environment
```bash
nvcc --version  # Should show CUDA version
nvidia-smi      # Should show your GPU
```

### 2. Start with Environment Check
```bash
nvcc -o check_env 1.1_check_gpu_environment.cu
./check_env
```

### 3. Run CPU Baseline
```bash
nvcc -o cpu_baseline 1.2_cpu_baseline_convolution.cu
./cpu_baseline
```

### 4. Test GPU Implementation
```bash
nvcc -O3 -o gpu_test 2.0_naive_convolution_complete.cu
./gpu_test
```

## Project Phases Overview

- **Phase 1**: Environment setup and CPU baseline
- **Phase 2**: Basic CUDA implementation  
- **Phase 3**: Shared memory optimization
- **Phase 4**: Kernel fusion techniques
- **Phase 5**: Advanced optimization and streams

## Expected Results

- **Phase 1**: Environment verification and performance baseline
- **Phase 2**: 2-5x speedup over CPU
- **Phase 3**: 5-10x speedup with shared memory
- **Phase 4**: 10-20x speedup with kernel fusion
- **Phase 5**: 15-25x speedup with final optimizations

## Troubleshooting

- **Compilation errors**: Check CUDA toolkit installation
- **Runtime errors**: Verify GPU compatibility and drivers
- **Performance issues**: Check block size and memory usage

## Next Steps

1. Read the full README.md for detailed explanations
2. Experiment with different block sizes and optimization flags
3. Try modifying kernels to understand the optimization process
4. Compare performance across different GPU architectures

## Need Help?

- Check CUDA documentation
- Review code comments in each file
- Test with smaller image sizes first
- Verify GPU memory availability 