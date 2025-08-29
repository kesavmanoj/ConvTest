#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void gpu_validation_kernel() {
    printf("GPU Thread %d in Block %d executing successfully\n", 
           threadIdx.x, blockIdx.x);
}

int main() {
    printf("=== CUDA Environment Verification ===\n");
    
    // Check CUDA runtime version
    int runtimeVersion;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get runtime version: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("CUDA Runtime Version: %d.%d\n", 
           runtimeVersion/1000, (runtimeVersion%100)/10);
    
    // Check CUDA driver version  
    int driverVersion;
    err = cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get driver version: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("CUDA Driver Version: %d.%d\n", 
           driverVersion/1000, (driverVersion%100)/10);
    
    // Enumerate GPU devices
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("\nDetected %d CUDA-capable device(s)\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found\n");
        return 1;
    }
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        
        printf("\n--- Device %d: %s ---\n", dev, deviceProp.name);
        printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / 1e9);
        printf("Shared Memory per Block: %d KB\n", (int)(deviceProp.sharedMemPerBlock / 1024));
        printf("L2 Cache Size: %d KB\n", (int)(deviceProp.l2CacheSize / 1024));
        printf("Memory Clock Rate: %.2f GHz\n", deviceProp.memoryClockRate * 1e-6);
        printf("Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);
        printf("Peak Memory Bandwidth: %.2f GB/s\n", 
               2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth/8) / 1e6);
        
        // Critical specifications for convolution optimization
        printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max Block Dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Max Grid Dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Multiprocessor Count: %d\n", deviceProp.multiProcessorCount);
        printf("Warp Size: %d\n", deviceProp.warpSize);
        printf("Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("Max Occupancy Support: %s\n", 
               (deviceProp.major >= 3) ? "Yes" : "Limited");
    }
    
    // Select first device for kernel execution
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Test basic kernel execution
    printf("\n=== GPU Kernel Execution Test ===\n");
    gpu_validation_kernel<<<2, 4>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        printf("✓ GPU kernel execution successful\n");
        return 0;
    } else {
        printf("✗ GPU kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
}
