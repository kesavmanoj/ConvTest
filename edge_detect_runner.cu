#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1); \
        } \
    } while (0)

// -----------------------------------------------------------------------------
// Edge kernels in constant memory (3x3)
// -----------------------------------------------------------------------------
__constant__ float c_kernel3x3[9];

static void select_kernel(const char* name, float out[9]) {
    if (!name) name = "laplace";
    if (strcmp(name, "laplace") == 0 || strcmp(name, "edge") == 0) {
        float k[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
        memcpy(out, k, sizeof(k));
        return;
    }
    if (strcmp(name, "sobelx") == 0) {
        float k[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        memcpy(out, k, sizeof(k));
        return;
    }
    if (strcmp(name, "sobely") == 0) {
        float k[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
        memcpy(out, k, sizeof(k));
        return;
    }
    // default
    float k[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
    memcpy(out, k, sizeof(k));
}

// -----------------------------------------------------------------------------
// Phase 2 style: global-memory conv (no fusion)
// -----------------------------------------------------------------------------
__global__ void conv3x3_global_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = x + kx;
            int py = y + ky;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                float v = input[py * width + px];
                float w = c_kernel3x3[(ky + 1) * 3 + (kx + 1)];
                sum += v * w;
            }
        }
    }
    output[y * width + x] = sum;
}

// -----------------------------------------------------------------------------
// Phase 4 style: conv + ReLU fusion (global-memory)
// -----------------------------------------------------------------------------
__global__ void conv3x3_global_relu_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = x + kx;
            int py = y + ky;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                float v = input[py * width + px];
                float w = c_kernel3x3[(ky + 1) * 3 + (kx + 1)];
                sum += v * w;
            }
        }
    }
    output[y * width + x] = fmaxf(0.0f, sum);
}

// -----------------------------------------------------------------------------
// Phase 5 style: shared-memory 16x16 tile with padding, conv + ReLU
// -----------------------------------------------------------------------------
#define TILE_X 16
#define TILE_Y 16
#define R 1
#define PAD 1
#define SMEM_W (TILE_X + 2*R + PAD)
#define SMEM_H (TILE_Y + 2*R)

__global__ void conv3x3_shared_relu_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int width, int height) {
    __shared__ float tile[SMEM_H][SMEM_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int out_x = bx * TILE_X + tx;
    int out_y = by * TILE_Y + ty;

    // cooperative load
    for (int ly = ty; ly < SMEM_H; ly += blockDim.y) {
        int gy = by * TILE_Y + ly - R;
        bool y_in = (gy >= 0 && gy < height);
        for (int lx = tx; lx < SMEM_W; lx += blockDim.x) {
            int gx = bx * TILE_X + lx - R;
            float v = 0.0f;
            if (y_in && gx >= 0 && gx < width) v = input[gy * width + gx];
            tile[ly][lx] = v;
        }
    }
    __syncthreads();

    if (out_x < width && out_y < height) {
        int sx = tx + R;
        int sy = ty + R;
        float sum = 0.0f;
        #pragma unroll
        for (int ky = -1; ky <= 1; ky++) {
            #pragma unroll
            for (int kx = -1; kx <= 1; kx++) {
                sum += tile[sy + ky][sx + kx] * c_kernel3x3[(ky + 1) * 3 + (kx + 1)];
            }
        }
        output[out_y * width + out_x] = fmaxf(0.0f, sum);
    }
}

// -----------------------------------------------------------------------------
// Simple I/O helpers for binary float32 images
// -----------------------------------------------------------------------------
static int read_f32(const char* path, float** data_out, size_t expected) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }
    float* buf = (float*)malloc(expected * sizeof(float));
    if (!buf) { fclose(f); return -2; }
    size_t n = fread(buf, sizeof(float), expected, f);
    fclose(f);
    if (n != expected) { free(buf); return -3; }
    *data_out = buf; return 0;
}

static int write_f32(const char* path, const float* data, size_t count) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen"); return -1; }
    size_t n = fwrite(data, sizeof(float), count, f);
    fclose(f);
    return n == count ? 0 : -2;
}

// -----------------------------------------------------------------------------
// Main: args
//   --mode=phase2|phase4|phase5
//   --kernel=edge|sobelx|sobely
//   --in=PATH --out=PATH --w=WIDTH --h=HEIGHT
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    const char* mode = "phase2";
    const char* kname = "edge";
    const char* in_path = NULL;
    const char* out_path = NULL;
    int width = 0, height = 0;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--mode=", 7) == 0) mode = argv[i] + 7;
        else if (strncmp(argv[i], "--kernel=", 9) == 0) kname = argv[i] + 9;
        else if (strncmp(argv[i], "--in=", 5) == 0) in_path = argv[i] + 5;
        else if (strncmp(argv[i], "--out=", 6) == 0) out_path = argv[i] + 6;
        else if (strncmp(argv[i], "--w=", 4) == 0) width = atoi(argv[i] + 4);
        else if (strncmp(argv[i], "--h=", 4) == 0) height = atoi(argv[i] + 4);
    }

    if (!in_path || !out_path || width <= 0 || height <= 0) {
        fprintf(stderr, "Usage: %s --mode=phase2|phase4|phase5 --kernel=edge|sobelx|sobely --in=PATH --out=PATH --w=W --h=H\n", argv[0]);
        return 1;
    }

    size_t pixels = (size_t)width * height;
    float* h_in = NULL;
    if (read_f32(in_path, &h_in, pixels) != 0) {
        fprintf(stderr, "Failed to read input float32 image: %s\n", in_path);
        return 2;
    }
    float* h_out = (float*)malloc(pixels * sizeof(float));
    if (!h_out) { free(h_in); return 3; }

    float k[9];
    select_kernel(kname, k);
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel3x3, k, 9 * sizeof(float)));

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, pixels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, pixels * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    if (strcmp(mode, "phase2") == 0) {
        conv3x3_global_kernel<<<grid, block>>>(d_in, d_out, width, height);
    } else if (strcmp(mode, "phase4") == 0) {
        conv3x3_global_relu_kernel<<<grid, block>>>(d_in, d_out, width, height);
    } else if (strcmp(mode, "phase5") == 0) {
        conv3x3_shared_relu_kernel<<<grid, block>>>(d_in, d_out, width, height);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_out); return 4;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    if (write_f32(out_path, h_out, pixels) != 0) {
        fprintf(stderr, "Failed to write output: %s\n", out_path);
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}


