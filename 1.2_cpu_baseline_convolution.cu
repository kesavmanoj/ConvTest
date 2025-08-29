#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// High-performance CPU convolution reference implementation
class CPUConvolutionBaseline {
private:
    float* input_image;
    float* output_image;
    float* convolution_kernel;
    int image_width, image_height, kernel_size;
    
public:
    CPUConvolutionBaseline(int width, int height, int kernel_dim) 
        : image_width(width), image_height(height), kernel_size(kernel_dim),
          input_image(nullptr), output_image(nullptr), convolution_kernel(nullptr) {
        
        size_t image_bytes = width * height * sizeof(float);
        size_t kernel_bytes = kernel_dim * kernel_dim * sizeof(float);
        
        input_image = (float*)aligned_alloc(32, image_bytes);
        if (!input_image) {
            fprintf(stderr, "Failed to allocate input image memory\n");
            return;
        }
        
        output_image = (float*)aligned_alloc(32, image_bytes);
        if (!output_image) {
            fprintf(stderr, "Failed to allocate output image memory\n");
            free(input_image);
            input_image = nullptr;
            return;
        }
        
        convolution_kernel = (float*)aligned_alloc(32, kernel_bytes);
        if (!convolution_kernel) {
            fprintf(stderr, "Failed to allocate kernel memory\n");
            free(input_image);
            free(output_image);
            input_image = nullptr;
            output_image = nullptr;
            return;
        }
        
        initialize_test_data();
    }
    
    ~CPUConvolutionBaseline() {
        if (input_image) free(input_image);
        if (output_image) free(output_image);
        if (convolution_kernel) free(convolution_kernel);
    }
    
    bool is_valid() const {
        return input_image && output_image && convolution_kernel;
    }
    
    void initialize_test_data() {
        if (!is_valid()) return;
        
        // Generate reproducible test pattern
        srand(42);
        for (int i = 0; i < image_width * image_height; i++) {
            input_image[i] = (float)rand() / RAND_MAX;
        }
        
        // Standard Gaussian blur kernel (3x3)
        if (kernel_size == 3) {
            float gaussian_kernel[9] = {
                1.0f/16, 2.0f/16, 1.0f/16,
                2.0f/16, 4.0f/16, 2.0f/16,
                1.0f/16, 2.0f/16, 1.0f/16
            };
            memcpy(convolution_kernel, gaussian_kernel, 9 * sizeof(float));
        }
        
        printf("Initialized %dx%d image with %dx%d kernel\n", 
               image_width, image_height, kernel_size, kernel_size);
    }
    
    double execute_cpu_convolution() {
        if (!is_valid()) return -1.0;
        
        #ifdef _WIN32
        LARGE_INTEGER frequency, start_time, end_time;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start_time);
        #else
        struct timeval start_time, end_time;
        gettimeofday(&start_time, NULL);
        #endif
        
        int kernel_radius = kernel_size / 2;
        
        // Optimized CPU convolution with loop unrolling
        for (int y = 0; y < image_height; y++) {
            for (int x = 0; x < image_width; x++) {
                float accumulator = 0.0f;
                
                // Manual loop unrolling for 3x3 kernel
                if (kernel_size == 3) {
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int py = y + ky;
                            int px = x + kx;
                            
                            if (px >= 0 && px < image_width && 
                                py >= 0 && py < image_height) {
                                accumulator += input_image[py * image_width + px] * 
                                             convolution_kernel[(ky + 1) * 3 + (kx + 1)];
                            }
                        }
                    }
                } else {
                    // General kernel size handling
                    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
                        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                            int py = y + ky;
                            int px = x + kx;
                            
                            if (px >= 0 && px < image_width && 
                                py >= 0 && py < image_height) {
                                accumulator += input_image[py * image_width + px] * 
                                             convolution_kernel[(ky + kernel_radius) * kernel_size + 
                                                              (kx + kernel_radius)];
                            }
                        }
                    }
                }
                
                output_image[y * image_width + x] = accumulator;
            }
        }
        
        #ifdef _WIN32
        QueryPerformanceCounter(&end_time);
        double execution_time = (double)(end_time.QuadPart - start_time.QuadPart) * 1000.0 / frequency.QuadPart;
        #else
        gettimeofday(&end_time, NULL);
        double execution_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                               (end_time.tv_usec - start_time.tv_usec) / 1000.0;
        #endif
        
        return execution_time;
    }
    
    void benchmark_cpu_performance(int iterations = 10) {
        printf("\n=== CPU Baseline Benchmark ===\n");
        
        double total_time = 0.0;
        double min_time = 1e9, max_time = 0.0;
        
        // Warm-up run
        execute_cpu_convolution();
        
        for (int i = 0; i < iterations; i++) {
            double iter_time = execute_cpu_convolution();
            total_time += iter_time;
            min_time = fmin(min_time, iter_time);
            max_time = fmax(max_time, iter_time);
        }
        
        double avg_time = total_time / iterations;
        double gflops = (image_width * image_height * kernel_size * kernel_size * 2.0 * 1e-6) / avg_time;
        
        printf("CPU Performance Results:\n");
        printf("  Average Time: %.2f ms\n", avg_time);
        printf("  Min Time: %.2f ms\n", min_time);
        printf("  Max Time: %.2f ms\n", max_time);
        printf("  Estimated GFLOPS: %.2f\n", gflops);
        printf("  Memory Bandwidth: %.2f GB/s\n", 
               (2 * image_width * image_height * sizeof(float) * 1e-6) / avg_time);
    }
    
    // Export data for GPU comparison
    float* get_input_data() { return input_image; }
    float* get_output_data() { return output_image; }
    float* get_kernel_data() { return convolution_kernel; }
};

int main() {
    printf("=== CUDA Convolution Baseline System Verification ===\n");
    
    // Test multiple image sizes for scalability analysis
    int test_sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int size = test_sizes[i];
        printf("\n--- Testing %dx%d Image ---\n", size, size);
        
        CPUConvolutionBaseline baseline(size, size, 3);
        if (!baseline.is_valid()) {
            fprintf(stderr, "Failed to initialize baseline for %dx%d image\n", size, size);
            continue;
        }
        baseline.benchmark_cpu_performance(5);
    }
    
    printf("\n✓ Baseline system verification complete\n");
    printf("✓ Ready to proceed to Phase 2: CUDA Implementation\n");
    
    return 0;
}
