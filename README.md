# ConvTest: CUDA Convolution Optimization Project

## Project Overview

ConvTest is a comprehensive CUDA project that demonstrates the evolution of 2D convolution optimization techniques on GPU hardware. The project is structured as a progressive learning path from basic GPU environment verification to advanced optimization techniques including shared memory usage, kernel fusion, and multi-stream pipelines.

**Note**: This project is a work-in-progress and may contain errors or incomplete implementations. It serves as a learning resource for CUDA optimization techniques.

## Project Structure

The project is organized into 5 phases, each building upon the previous one:

### Phase 1: Environment Setup and Baseline
- **`1.1_check_gpu_environment.cu`** - GPU environment verification and device capabilities analysis
- **`1.2_cpu_baseline_convolution.cu`** - High-performance CPU convolution reference implementation
- **`1.3_cuda_optimization_setup.cu`** - CUDA environment optimization and memory hierarchy analysis

### Phase 2: Basic CUDA Implementation
- **`2.0_naive_convolution_complete.cu`** - Baseline CUDA convolution implementation with performance benchmarking

### Phase 3: Shared Memory Optimization
- **`3.0_phase3_final_corrected.cu`** - Multiple shared memory optimization approaches with correctness testing

### Phase 4: Kernel Fusion
- **`4.0_phase4_kernel_fusion.cu`** - Advanced kernel fusion techniques combining multiple operations

### Phase 5: Final Optimization
- **`5.0_phase5_final_optimization.cu`** - Multi-stream pipelines and final optimization techniques

## Detailed File Descriptions

### 1.1_check_gpu_environment.cu
**Purpose**: GPU environment verification and device capabilities analysis

**Features**:
- CUDA runtime and driver version detection
- GPU device enumeration with detailed specifications
- Memory hierarchy analysis (L1, L2, global memory)
- Compute capability and occupancy analysis
- Basic kernel execution test

**Use Case**: First step to verify CUDA environment is properly configured and understand target hardware capabilities.

### 1.2_cpu_baseline_convolution.cu
**Purpose**: High-performance CPU convolution reference implementation

**Features**:
- Optimized CPU convolution with loop unrolling
- Memory-aligned allocation for performance
- Gaussian blur kernel (3x3) implementation
- Comprehensive benchmarking with multiple image sizes
- Performance metrics (GFLOPS, memory bandwidth)

**Use Case**: Establishes baseline performance for comparison with GPU implementations.

### 1.3_cuda_optimization_setup.cu
**Purpose**: CUDA environment optimization and memory hierarchy analysis

**Features**:
- Optimal device configuration for convolution workloads
- Shared memory bank conflict optimization
- Memory mapping configuration
- Tile size recommendations based on shared memory constraints
- Theoretical occupancy calculations

**Use Case**: Prepares CUDA environment for optimal convolution performance.

### 2.0_naive_convolution_complete.cu
**Purpose**: Baseline CUDA convolution implementation with performance benchmarking

**Features**:
- Naive 2D convolution kernel
- Optimized 3x3 kernel with manual unrolling
- CUDA timing utilities
- CPU reference for validation
- Comprehensive performance analysis
- Memory bandwidth calculations

**Use Case**: Establishes GPU baseline performance and validates correctness against CPU reference.

### 3.0_phase3_final_corrected.cu
**Purpose**: Multiple shared memory optimization approaches with correctness testing

**Features**:
- Three different shared memory implementations:
  - Simple working approach
  - Ultra-safe with corrected boundary handling
  - Bulletproof direct mapping approach
- Comprehensive correctness testing
- Performance comparison between implementations
- Boundary condition handling for shared memory

**Use Case**: Demonstrates shared memory optimization techniques and their impact on performance and correctness.

### 4.0_phase4_kernel_fusion.cu
**Purpose**: Advanced kernel fusion techniques combining multiple operations

**Features**:
- Fused convolution + ReLU activation
- Fused convolution + ReLU + max pooling
- Dual convolution with feature combination
- Baseline separate kernel implementations for comparison
- Memory traffic analysis
- Performance benchmarking of fusion benefits

**Use Case**: Shows how kernel fusion can dramatically improve performance by reducing memory transfers and kernel launch overhead.

### 5.0_phase5_final_optimization.cu
**Purpose**: Multi-stream pipelines and final optimization techniques

**Features**:
- Optimized 16x16 block size for perfect warp efficiency
- Multi-stream pipeline implementation
- Occupancy analysis for different block configurations
- Advanced timing with stream support
- Final performance optimization

**Use Case**: Demonstrates advanced CUDA optimization techniques including stream management and occupancy optimization.

## Key Learning Objectives

1. **GPU Environment Setup**: Understanding CUDA device capabilities and configuration
2. **Baseline Implementation**: Establishing performance baselines for comparison
3. **Shared Memory Optimization**: Learning to use shared memory effectively
4. **Kernel Fusion**: Combining multiple operations for better performance
5. **Advanced Optimization**: Multi-stream pipelines and occupancy analysis

## Technical Highlights

### Memory Optimization
- Shared memory usage for data reuse
- Constant memory for kernel coefficients
- Memory coalescing considerations
- Bank conflict avoidance

### Performance Techniques
- Loop unrolling
- Block size optimization
- Kernel fusion
- Multi-stream execution
- Occupancy maximization

### Correctness and Validation
- CPU reference implementations
- Comprehensive error checking
- Boundary condition handling
- Performance validation

## Compilation and Usage

### Prerequisites
- CUDA Toolkit (version 11.0 or later recommended)
- NVIDIA GPU with compute capability 3.0 or higher
- C++ compiler (GCC, Clang, or MSVC). On Windows, open an "x64 Native Tools Command Prompt for VS" or ensure your MSVC toolchain is on PATH.

### Compilation
Note: Adjust the architecture flag to your GPU (e.g., `-arch=sm_75` for Tesla T4, `-arch=sm_86` for RTX 30xx).

```bash
# Linux / WSL
# Phase 1
nvcc -O2 -arch=sm_75 -o 1.1_check_gpu_environment 1.1_check_gpu_environment.cu
nvcc -O2 -arch=sm_75 -o 1.2_cpu_baseline_convolution 1.2_cpu_baseline_convolution.cu
nvcc -O2 -arch=sm_75 -o 1.3_cuda_optimization_setup 1.3_cuda_optimization_setup.cu

# Phase 2 (filename corrected)
nvcc -O3 -arch=sm_75 -o 2.0_naive_convolution_complete 2.0_naive_convolution_complete.cu

# Phase 3
nvcc -O3 -arch=sm_75 -o 3.0_phase3_final_corrected 3.0_phase3_final_corrected.cu

# Phase 4
nvcc -O3 -arch=sm_75 -o 4.0_phase4_kernel_fusion 4.0_phase4_kernel_fusion.cu

# Phase 5 (output name aligned with file)
nvcc -O3 -arch=sm_75 -o 5.0_phase5_final_optimization 5.0_phase5_final_optimization.cu
```

```powershell
# Windows (PowerShell)
# Phase 1
nvcc -O2 -arch=sm_75 -o 1.1_check_gpu_environment.exe 1.1_check_gpu_environment.cu
nvcc -O2 -arch=sm_75 -o 1.2_cpu_baseline_convolution.exe 1.2_cpu_baseline_convolution.cu
nvcc -O2 -arch=sm_75 -o 1.3_cuda_optimization_setup.exe 1.3_cuda_optimization_setup.cu

# Phase 2 (filename corrected)
nvcc -O3 -arch=sm_75 -o 2.0_naive_convolution_complete.exe 2.0_naive_convolution_complete.cu

# Phase 3
nvcc -O3 -arch=sm_75 -o 3.0_phase3_final_corrected.exe 3.0_phase3_final_corrected.cu

# Phase 4
nvcc -O3 -arch=sm_75 -o 4.0_phase4_kernel_fusion.exe 4.0_phase4_kernel_fusion.cu

# Phase 5 (output name aligned with file)
nvcc -O3 -arch=sm_75 -o 5.0_phase5_final_optimization.exe 5.0_phase5_final_optimization.cu
```

### Execution
```bash
# Linux / WSL
./1.1_check_gpu_environment
./1.2_cpu_baseline_convolution
./1.3_cuda_optimization_setup
./2.0_naive_convolution_complete
./3.0_phase3_final_corrected
./4.0_phase4_kernel_fusion
./5.0_phase5_final_optimization
```

```powershell
# Windows (PowerShell)
./1.1_check_gpu_environment.exe
./1.2_cpu_baseline_convolution.exe
./1.3_cuda_optimization_setup.exe
./2.0_naive_convolution_complete.exe
./3.0_phase3_final_corrected.exe
./4.0_phase4_kernel_fusion.exe
./5.0_phase5_final_optimization.exe
```

## Performance Expectations

### Tesla T4 Target Performance
- **Phase 1**: Environment verification and baseline establishment
- **Phase 2**: 2-5x speedup over CPU baseline
- **Phase 3**: 5-10x speedup with shared memory optimization
- **Phase 4**: 10-20x speedup with kernel fusion
- **Phase 5**: 15-25x speedup with final optimizations

### Memory Bandwidth Utilization
- Target: 80-90% of peak memory bandwidth
- Shared memory usage: 16–18 KB per block (phase-dependent)
- Optimal block size: 16×16 (256 threads, 8 warps) as emphasized in Phases 4 and 5

## Known Issues and Limitations

1. **Project Status**: This is a work-in-progress project with potential errors
2. **Boundary Conditions**: Some earlier implementations may have edge case handling issues; later phases include corrected indexing
3. **Memory Management**: Error handling could be improved in some areas
4. **Validation**: Some test cases may not cover all scenarios

## Future Improvements

1. **Error Handling**: More robust CUDA error checking
2. **Testing**: Comprehensive unit tests for all implementations
3. **Documentation**: Inline code documentation and comments
4. **Performance**: Additional optimization techniques (e.g., tensor cores when available)
5. **Flexibility**: Support for different kernel sizes and data types

## Contributing

This project serves as a learning resource. Contributions to improve correctness, performance, or documentation are welcome.

## License

This project is provided as-is for educational purposes. Use at your own risk.

## Contact

For questions or issues related to this project, please refer to the code comments and CUDA documentation for guidance. If you are targeting a GPU other than Tesla T4, make sure to adjust `-arch=sm_XX` accordingly.