#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Fixed format specifiers and removed deprecated calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Naive 2D convolution kernel - baseline implementation
__global__ void naive_convolution_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output, 
    const float* __restrict__ kernel,
    int image_width,
    int image_height, 
    int kernel_size) {
    
    // Calculate global thread coordinates using proper indexing
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check - critical for correctness
    if (global_x >= image_width || global_y >= image_height) {
        return;
    }
    
    int kernel_radius = kernel_size / 2;
    float accumulator = 0.0f;
    
    // Convolution computation - each thread processes one output pixel
    // This creates high global memory access redundancy (our optimization target)
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            
            int input_x = global_x + kx;
            int input_y = global_y + ky;
            
            // Zero padding for boundary handling
            if (input_x >= 0 && input_x < image_width && 
                input_y >= 0 && input_y < image_height) {
                
                // Global memory accesses - will be optimized in Phase 3
                float input_value = input[input_y * image_width + input_x];
                float kernel_weight = kernel[(ky + kernel_radius) * kernel_size + 
                                            (kx + kernel_radius)];
                
                accumulator += input_value * kernel_weight;
            }
        }
    }
    
    // Write result - single global memory write per thread
    output[global_y * image_width + global_x] = accumulator;
}

// Optimized 3x3 kernel with manual unrolling
__global__ void naive_convolution_3x3_unrolled(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ kernel,
    int image_width,
    int image_height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= image_width || y >= image_height) return;
    
    float sum = 0.0f;
    
    // Manual unrolling reduces loop overhead
    if (x >= 1 && y >= 1 && x < image_width - 1 && y < image_height - 1) {
        // Interior pixels - no boundary checks needed
        sum += input[(y-1) * image_width + (x-1)] * kernel[0];
        sum += input[(y-1) * image_width + x] * kernel[1];  
        sum += input[(y-1) * image_width + (x+1)] * kernel[2];
        sum += input[y * image_width + (x-1)] * kernel[3];
        sum += input[y * image_width + x] * kernel[4];
        sum += input[y * image_width + (x+1)] * kernel[5];
        sum += input[(y+1) * image_width + (x-1)] * kernel[6];
        sum += input[(y+1) * image_width + x] * kernel[7];
        sum += input[(y+1) * image_width + (x+1)] * kernel[8];
    } else {
        // Boundary pixels require explicit checks
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = x + kx;
                int py = y + ky;
                
                if (px >= 0 && px < image_width && py >= 0 && py < image_height) {
                    sum += input[py * image_width + px] * 
                           kernel[(ky + 1) * 3 + (kx + 1)];
                }
            }
        }
    }
    
    output[y * image_width + x] = sum;
}

// CUDA timing utility
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }
};

// Host wrapper function for kernel execution
float launch_naive_convolution(
    float* h_input,
    float* h_output,
    float* h_kernel,
    int width, int height, int kernel_size,
    int block_x = 16, int block_y = 16) {
    
    // Memory allocation
    size_t image_bytes = width * height * sizeof(float);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    
    float *d_input, *d_output, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
    
    // Data transfer to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));
    
    // Configure launch parameters
    dim3 block_size(block_x, block_y);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );
    
    printf("Naive Convolution: Grid(%d,%d) Block(%d,%d)\n",
           grid_size.x, grid_size.y, block_size.x, block_size.y);
    
    // Time kernel execution
    CudaTimer timer;
    timer.start();
    
    // Launch appropriate kernel
    if (kernel_size == 3) {
        naive_convolution_3x3_unrolled<<<grid_size, block_size>>>(
            d_input, d_output, d_kernel, width, height);
    } else {
        naive_convolution_2d_kernel<<<grid_size, block_size>>>(
            d_input, d_output, d_kernel, width, height, kernel_size);
    }
    
    CUDA_CHECK(cudaGetLastError());
    float execution_time = timer.stop();
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    return execution_time;
}

// CPU reference for validation
void cpu_convolution_reference(
    const float* input, float* output, const float* kernel,
    int width, int height, int kernel_size) {
    
    int radius = kernel_size / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        sum += input[py * width + px] * 
                               kernel[(ky + radius) * kernel_size + (kx + radius)];
                    }
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

// Validation function
bool validate_results(float* gpu_output, float* cpu_output, 
                     int width, int height, float tolerance = 1e-5f) {
    
    int num_pixels = width * height;
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < num_pixels; i++) {
        float error = fabsf(gpu_output[i] - cpu_output[i]);
        if (error > max_error) max_error = error;
        if (error > tolerance) errors++;
    }
    
    printf("Validation: %d errors, max error: %.2e\n", errors, max_error);
    return errors == 0;
}

// Comprehensive benchmark function
void benchmark_naive_implementation() {
    printf("\n=== Phase 2: Naive Convolution Benchmark ===\n");
    
    // Test configurations matching your CPU baseline
    int test_sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    // Standard 3x3 Gaussian blur kernel
    float gaussian_kernel[9] = {
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    };
    
    printf("GPU Device Info:\n");
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("  Device: %s\n", prop.name);
    printf("  Peak Memory Bandwidth: %.1f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1e6);
    
    for (int i = 0; i < num_sizes; i++) {
        int size = test_sizes[i];
        printf("\n--- %dx%d Image Benchmark ---\n", size, size);
        
        // Allocate memory
        size_t image_size = size * size;
        float* h_input = (float*)malloc(image_size * sizeof(float));
        float* h_gpu_output = (float*)malloc(image_size * sizeof(float));
        float* h_cpu_output = (float*)malloc(image_size * sizeof(float));
        
        // Initialize with same pattern as CPU baseline
        srand(42);
        for (size_t j = 0; j < image_size; j++) {
            h_input[j] = (float)rand() / RAND_MAX;
        }
        
        // CPU reference timing
        #ifdef _WIN32
        LARGE_INTEGER frequency, start, end;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);
        cpu_convolution_reference(h_input, h_cpu_output, gaussian_kernel, 
                                 size, size, 3);
        QueryPerformanceCounter(&end);
        double cpu_time = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
        #else
        struct timeval start, end;
        gettimeofday(&start, NULL);
        cpu_convolution_reference(h_input, h_cpu_output, gaussian_kernel, 
                                 size, size, 3);
        gettimeofday(&end, NULL);
        double cpu_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                         (end.tv_usec - start.tv_usec) / 1000.0;
        #endif
        
        // GPU kernel execution - multiple iterations for accurate timing
        const int iterations = 20;
        float total_gpu_time = 0.0f;
        
        // Warm-up run
        launch_naive_convolution(h_input, h_gpu_output, gaussian_kernel, 
                               size, size, 3);
        
        // Benchmark iterations
        for (int iter = 0; iter < iterations; iter++) {
            float iter_time = launch_naive_convolution(h_input, h_gpu_output, 
                                                     gaussian_kernel, size, size, 3);
            total_gpu_time += iter_time;
        }
        
        float avg_gpu_time = total_gpu_time / iterations;
        
        // Validate correctness
        bool results_valid = validate_results(h_gpu_output, h_cpu_output, size, size);
        
        // Performance analysis
        size_t total_ops = image_size * 9 * 2; // 9 multiply-adds per pixel
        float gpu_gflops = (total_ops * 1e-6f) / avg_gpu_time;
        float cpu_gflops = (total_ops * 1e-6f) / cpu_time;
        
        // Memory analysis
        size_t memory_reads = image_size * 9 + 9; // Read overlapping regions + kernel
        size_t memory_writes = image_size;
        size_t total_memory = (memory_reads + memory_writes) * sizeof(float);
        float bandwidth = (total_memory * 1e-6f) / avg_gpu_time;
        
        // Calculate speedup
        float speedup = cpu_time / avg_gpu_time;
        
        printf("Performance Results:\n");
        printf("  CPU Time: %.2f ms (%.2f GFLOPS)\n", cpu_time, cpu_gflops);
        printf("  GPU Time: %.2f ms (%.2f GFLOPS)\n", avg_gpu_time, gpu_gflops);
        printf("  Speedup: %.2fx\n", speedup);
        printf("  Memory Bandwidth: %.1f GB/s\n", bandwidth);
        printf("  Memory Efficiency: %.1f%% of peak\n", 
               (bandwidth / 320.1f) * 100.0f); // Your T4 peak
        printf("  Validation: %s\n", results_valid ? "PASSED" : "FAILED");
        
        // Cleanup
        free(h_input);
        free(h_gpu_output);
        free(h_cpu_output);
    }
    
    printf("\n✓ Phase 2 baseline established\n");
    printf("✓ Ready for Phase 3: Shared Memory Optimization\n");
}

int main() {
    printf("=== Phase 2: Baseline CUDA Implementation ===\n");
    printf("Tesla T4 Detected - Targeting 12× optimization in Phase 3\n");
    
    benchmark_naive_implementation();
    
    return 0;
}
