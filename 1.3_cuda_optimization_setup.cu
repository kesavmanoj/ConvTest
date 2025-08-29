#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

class CUDAEnvironmentOptimizer {
public:
    static bool configure_optimal_device() {
        int device_id;
        cudaError_t err = cudaGetDevice(&device_id);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Enable optimal caching behavior for convolution
        err = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to set cache config: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Note: cudaDeviceSetSharedMemConfig is deprecated, so we skip it
        // The default 4-byte bank configuration is usually optimal anyway
        
        // Set device flags for mapped memory optimization
        err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | 
                                cudaDeviceMapHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to set device flags: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        printf("✓ CUDA device optimization configured\n");
        printf("✓ Shared memory preference: Enabled\n");
        printf("✓ Bank conflict optimization: Using default 4-byte banks\n");
        printf("✓ Memory mapping: Enabled\n");
        return true;
    }
    
    static bool analyze_memory_hierarchy() {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        printf("\n=== Memory Hierarchy Analysis ===\n");
        printf("L1 Cache/Shared Memory: %d KB (configurable)\n", 
               (int)(prop.sharedMemPerBlock / 1024));
        printf("L2 Cache: %d KB (hardware)\n", 
               (int)(prop.l2CacheSize / 1024));
        printf("Global Memory: %.1f GB\n", 
               prop.totalGlobalMem / 1e9);
        printf("Memory Clock: %.2f GHz\n", 
               prop.memoryClockRate * 1e-6);
        printf("Memory Bandwidth: %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1e6);
               
        // Calculate optimal tile sizes for shared memory
        int max_shared_mem = prop.sharedMemPerBlock;
        int optimal_tile_size = (int)sqrt(max_shared_mem / (4 * sizeof(float))); // Conservative estimate
        printf("Recommended max tile size: %dx%d\n", optimal_tile_size, optimal_tile_size);
        
        // Calculate theoretical occupancy limits
        int max_threads_per_block = prop.maxThreadsPerBlock;
        if (max_threads_per_block > 0) {
            int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / max_threads_per_block;
            printf("Max theoretical occupancy: %d blocks per SM\n", max_blocks_per_sm);
        } else {
            printf("Max theoretical occupancy: Unable to calculate\n");
        }
        return true;
    }
};

int main() {
    printf("=== Advanced CUDA Environment Setup ===\n");
    
    // Check if CUDA is available
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA not available: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    // Select first device
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    if (!CUDAEnvironmentOptimizer::configure_optimal_device()) {
        fprintf(stderr, "Failed to configure optimal device\n");
        return 1;
    }
    
    if (!CUDAEnvironmentOptimizer::analyze_memory_hierarchy()) {
        fprintf(stderr, "Failed to analyze memory hierarchy\n");
        return 1;
    }
    
    // Test CUDA compilation with optimization flags
    printf("\n✓ Environment setup complete\n");
    printf("✓ Ready for GPU kernel development\n");
    
    return 0;
}
