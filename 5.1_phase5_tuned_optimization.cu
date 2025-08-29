#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// Tuned Phase 5 (v2): 16x16 tile (256 threads), coalesced cooperative load,
// padding, restrict qualifiers, and read-only cache hints.
// Motivation: maintain 256 threads (8 warps), improve coalescing, reduce
// shared-memory bank conflicts, and potentially raise occupancy.
// -----------------------------------------------------------------------------

#define TILE_X 16
#define TILE_Y 16
#define KERNEL_RADIUS 1
#define PAD 1                         // padding to mitigate bank conflicts with smaller footprint

// Shared tile dimensions (including halo + padding along X)
#define SMEM_W (TILE_X + 2 * KERNEL_RADIUS + PAD)   // 16 + 2 + PAD
#define SMEM_H (TILE_Y + 2 * KERNEL_RADIUS)         // 16 + 2

__constant__ float c_kernel3x3[9];

#ifndef USE_LDG
#define USE_LDG 1
#endif

// Kernel: tuned shared-memory convolution (3x3) with 32x8 layout
__global__ void conv3x3_shared_16x16(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int width, int height) {
    __shared__ float tile[SMEM_H][SMEM_W];

    const int tx = threadIdx.x;   // [0..31]
    const int ty = threadIdx.y;   // [0..7]
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int out_x = bx * TILE_X + tx;
    const int out_y = by * TILE_Y + ty;

    // Cooperative load of the SMEM tile: cover (SMEM_H x SMEM_W) using 32x8 threads
    const int threads_x = blockDim.x; // 16
    const int threads_y = blockDim.y; // 16
    const int elements_x = SMEM_W;    // 16+2+PAD
    const int elements_y = SMEM_H;    // 16+2

    for (int ly = ty; ly < elements_y; ly += threads_y) {
        int gy = by * TILE_Y + ly - KERNEL_RADIUS;
        bool y_in = (gy >= 0 && gy < height);

        // Vectorized loads in chunks of 4 when possible
        int lx = tx * 4;
        for (; lx + 3 < elements_x; lx += threads_x * 4) {
            int gx = bx * TILE_X + lx - KERNEL_RADIUS;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (y_in && gx >= 0 && gx + 3 < width && ((gx & 3) == 0)) {
                const float* base = &input[gy * width + gx];
#if USE_LDG && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
                v.x = __ldg(base + 0);
                v.y = __ldg(base + 1);
                v.z = __ldg(base + 2);
                v.w = __ldg(base + 3);
#else
                v = *reinterpret_cast<const float4*>(base);
#endif
            } else {
                for (int k = 0; k < 4; ++k) {
                    int gxk = gx + k;
                    if (y_in && gxk >= 0 && gxk < width) {
#if USE_LDG && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
                        (&v.x)[k] = __ldg(&input[gy * width + gxk]);
#else
                        (&v.x)[k] = input[gy * width + gxk];
#endif
                    }
                }
            }
            tile[ly][lx + 0] = v.x;
            tile[ly][lx + 1] = v.y;
            tile[ly][lx + 2] = v.z;
            tile[ly][lx + 3] = v.w;
        }

        // Tail columns (scalar)
        for (; lx < elements_x; lx += threads_x) {
            int gx = bx * TILE_X + lx - KERNEL_RADIUS;
            float val = 0.f;
            if (y_in && gx >= 0 && gx < width) {
#if USE_LDG && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
                val = __ldg(&input[gy * width + gx]);
#else
                val = input[gy * width + gx];
#endif
            }
            tile[ly][lx] = val;
        }
    }

    __syncthreads();

    if (out_x < width && out_y < height) {
        // Position within the SMEM tile for the output pixel's (0,0) kernel tap
        const int sx = tx + KERNEL_RADIUS;
        const int sy = ty + KERNEL_RADIUS;

        float sum = 0.0f;
        #pragma unroll
        for (int ky = 0; ky < 3; ky++) {
            #pragma unroll
            for (int kx = 0; kx < 3; kx++) {
                sum += tile[sy + ky - 1][sx + kx - 1] * c_kernel3x3[ky * 3 + kx];
            }
        }
        sum = fmaxf(0.0f, sum);
        output[out_y * width + out_x] = sum;
    }
}

// Reference baseline (global memory, 16x16 launch but no smem)
__global__ void baseline_conv3x3_global(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
#if USE_LDG && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
        const float a0 = __ldg(&input[(y - 1) * width + (x - 1)]);
        const float a1 = __ldg(&input[(y - 1) * width + (x    )]);
        const float a2 = __ldg(&input[(y - 1) * width + (x + 1)]);
        const float b0 = __ldg(&input[(y    ) * width + (x - 1)]);
        const float b1 = __ldg(&input[(y    ) * width + (x    )]);
        const float b2 = __ldg(&input[(y    ) * width + (x + 1)]);
        const float c0 = __ldg(&input[(y + 1) * width + (x - 1)]);
        const float c1 = __ldg(&input[(y + 1) * width + (x    )]);
        const float c2 = __ldg(&input[(y + 1) * width + (x + 1)]);
#else
        const float a0 = input[(y - 1) * width + (x - 1)];
        const float a1 = input[(y - 1) * width + (x    )];
        const float a2 = input[(y - 1) * width + (x + 1)];
        const float b0 = input[(y    ) * width + (x - 1)];
        const float b1 = input[(y    ) * width + (x    )];
        const float b2 = input[(y    ) * width + (x + 1)];
        const float c0 = input[(y + 1) * width + (x - 1)];
        const float c1 = input[(y + 1) * width + (x    )];
        const float c2 = input[(y + 1) * width + (x + 1)];
#endif
        sum += a0 * c_kernel3x3[0];
        sum += a1 * c_kernel3x3[1];
        sum += a2 * c_kernel3x3[2];
        sum += b0 * c_kernel3x3[3];
        sum += b1 * c_kernel3x3[4];
        sum += b2 * c_kernel3x3[5];
        sum += c0 * c_kernel3x3[6];
        sum += c1 * c_kernel3x3[7];
        sum += c2 * c_kernel3x3[8];
    }
    output[y * width + x] = sum;
}

// Simple CUDA timer
class CudaTimer {
public:
    cudaEvent_t start_event, stop_event;
    CudaTimer() { CUDA_CHECK(cudaEventCreate(&start_event)); CUDA_CHECK(cudaEventCreate(&stop_event)); }
    ~CudaTimer(){ cudaEventDestroy(start_event); cudaEventDestroy(stop_event);}    
    void start(cudaStream_t s=0){ CUDA_CHECK(cudaEventRecord(start_event, s)); }
    float stop(cudaStream_t s=0){ CUDA_CHECK(cudaEventRecord(stop_event, s)); CUDA_CHECK(cudaEventSynchronize(stop_event)); float ms=0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event)); return ms; }
};

static void init_input(float* h, int n, unsigned seed){
    srand(seed);
    for (int i = 0; i < n; i++) {
        h[i] = (float)rand() / RAND_MAX;
    }
}

static void run_size(int size) {
    const int width = size;
    const int height = size;
    const size_t elems = (size_t)width * height;
    const size_t bytes = elems * sizeof(float);

    // Gaussian 3x3
    float hkernel[9] = {
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    };
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel3x3, hkernel, 9*sizeof(float)));

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    init_input(h_in, (int)elems, 42 + size);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 blockSm(TILE_X, TILE_Y);
    dim3 gridSm((width + TILE_X - 1)/TILE_X, (height + TILE_Y - 1)/TILE_Y);

    dim3 blockBase(TILE_X, TILE_Y);
    dim3 gridBase((width + TILE_X - 1)/TILE_X, (height + TILE_Y - 1)/TILE_Y);

    // Warm-up
    conv3x3_shared_16x16<<<gridSm, blockSm>>>(d_in, d_out, width, height);
    baseline_conv3x3_global<<<gridBase, blockBase>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel-only timing
    const int iters = 80;
    CudaTimer t1, t2;

    float total_shared = 0.0f;
    for (int i = 0; i < iters; i++) {
        t1.start();
        conv3x3_shared_16x16<<<gridSm, blockSm>>>(d_in, d_out, width, height);
        CUDA_CHECK(cudaGetLastError());
        total_shared += t1.stop();
    }

    float total_base = 0.0f;
    for (int i = 0; i < iters; i++) {
        t2.start();
        baseline_conv3x3_global<<<gridBase, blockBase>>>(d_in, d_out, width, height);
        CUDA_CHECK(cudaGetLastError());
        total_base += t2.stop();
    }

    float ms_shared = total_shared / iters;
    float ms_base = total_base / iters;

    // Simple GFLOPS estimate: 18 ops/pixel (9 mul + 9 add)
    double ops = (double)elems * 18.0;
    double gflops_shared = (ops * 1e-6) / ms_shared;
    double gflops_base = (ops * 1e-6) / ms_base;

    printf("\n--- %dx%d Tuned Benchmark (16x16 tile) ---\n", size, size);
    printf("Baseline (global, 16x16): %.3f ms (%.1f GFLOPS)\n", ms_base, (float)gflops_base);
    printf("Shared (16x16, padded): %.3f ms (%.1f GFLOPS)\n", ms_shared, (float)gflops_shared);
    printf("Speedup (shared/base): %.2fx\n", ms_base / ms_shared);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}

int main(){
    printf("=== Phase 5 Tuned v2: 16x16 Tile, Coalesced Loads, Padded SMEM, LDG ===\n");
    int sizes[] = {256, 512, 1024};
    for (int i = 0; i < 3; i++) run_size(sizes[i]);
    printf("\nGuidance: This tuned variant targets higher occupancy and lower SMEM pressure.\n");
    printf("Validate with Nsight Compute: occupancy, smem bank conflicts, ld/st coalescing.\n");
    return 0;
}


