#include "gpu_convolution.cuh"
#include "utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <algorithm>

// ============================================================================
// Constants
// ============================================================================

#define MAX_KERNEL_SIZE 7
#define MAX_KERNEL_ELEMENTS (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE)

// Constant memory for kernel coefficients
__constant__ float d_kernel[MAX_KERNEL_ELEMENTS];

// ============================================================================
// CUDA Device Info
// ============================================================================

void print_cuda_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    std::cout << "CUDA Devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
}

bool cuda_available() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

int get_cuda_device_count() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

// ============================================================================
// CUDA Kernel: Global Memory Version (Baseline)
// ============================================================================

__global__ void convolve_kernel_global(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width, int height, int channels,
    const float* __restrict__ kernel,
    int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half = kernel_size / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                int img_idx = (py * width + px) * channels + c;
                int k_idx = (ky + half) * kernel_size + (kx + half);
                
                sum += static_cast<float>(input[img_idx]) * kernel[k_idx];
            }
        }
        
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = static_cast<uint8_t>(fminf(fmaxf(sum, 0.0f), 255.0f));
    }
}

// ============================================================================
// CUDA Kernel: Constant Memory Version
// ============================================================================

__global__ void convolve_kernel_constant(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width, int height, int channels,
    int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half = kernel_size / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                int img_idx = (py * width + px) * channels + c;
                int k_idx = (ky + half) * kernel_size + (kx + half);
                
                sum += static_cast<float>(input[img_idx]) * d_kernel[k_idx];
            }
        }
        
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = static_cast<uint8_t>(fminf(fmaxf(sum, 0.0f), 255.0f));
    }
}

// ============================================================================
// CUDA Kernel: Shared Memory Version
// ============================================================================

template<int BLOCK_SIZE, int KERNEL_RADIUS>
__global__ void convolve_kernel_shared(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width, int height, int channels,
    int kernel_size)
{
    const int TILE_SIZE = BLOCK_SIZE + 2 * KERNEL_RADIUS;
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;
    
    int half = kernel_size / 2;
    
    for (int c = 0; c < channels; c++) {
        // Load center
        int shared_x = tx + KERNEL_RADIUS;
        int shared_y = ty + KERNEL_RADIUS;
        
        if (x < width && y < height) {
            tile[shared_y][shared_x] = static_cast<float>(
                input[(y * width + x) * channels + c]);
        } else {
            tile[shared_y][shared_x] = 0.0f;
        }
        
        // Load halo regions
        if (tx < KERNEL_RADIUS) {
            int load_x = max(0, (int)(blockIdx.x * BLOCK_SIZE) - KERNEL_RADIUS + tx);
            int load_y = min(max(y, 0), height - 1);
            tile[shared_y][tx] = static_cast<float>(
                input[(load_y * width + load_x) * channels + c]);
        }
        if (tx >= BLOCK_SIZE - KERNEL_RADIUS) {
            int load_x = min((int)(blockIdx.x * BLOCK_SIZE + BLOCK_SIZE + tx - (BLOCK_SIZE - KERNEL_RADIUS)), width - 1);
            int load_y = min(max(y, 0), height - 1);
            tile[shared_y][BLOCK_SIZE + KERNEL_RADIUS + tx - (BLOCK_SIZE - KERNEL_RADIUS)] = 
                static_cast<float>(input[(load_y * width + load_x) * channels + c]);
        }
        if (ty < KERNEL_RADIUS) {
            int load_x = min(max(x, 0), width - 1);
            int load_y = max(0, (int)(blockIdx.y * BLOCK_SIZE) - KERNEL_RADIUS + ty);
            tile[ty][shared_x] = static_cast<float>(
                input[(load_y * width + load_x) * channels + c]);
        }
        if (ty >= BLOCK_SIZE - KERNEL_RADIUS) {
            int load_x = min(max(x, 0), width - 1);
            int load_y = min((int)(blockIdx.y * BLOCK_SIZE + BLOCK_SIZE + ty - (BLOCK_SIZE - KERNEL_RADIUS)), height - 1);
            tile[BLOCK_SIZE + KERNEL_RADIUS + ty - (BLOCK_SIZE - KERNEL_RADIUS)][shared_x] = 
                static_cast<float>(input[(load_y * width + load_x) * channels + c]);
        }
        
        __syncthreads();
        
        if (x < width && y < height) {
            float sum = 0.0f;
            
            #pragma unroll
            for (int ky = -half; ky <= half; ky++) {
                #pragma unroll
                for (int kx = -half; kx <= half; kx++) {
                    int k_idx = (ky + half) * kernel_size + (kx + half);
                    sum += tile[ty + KERNEL_RADIUS + ky][tx + KERNEL_RADIUS + kx] * d_kernel[k_idx];
                }
            }
            
            output[(y * width + x) * channels + c] = 
                static_cast<uint8_t>(fminf(fmaxf(sum, 0.0f), 255.0f));
        }
        
        __syncthreads();
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void convolve_cuda_global(const uint8_t* input, uint8_t* output,
                          int width, int height, int channels,
                          const float* kernel, int kernel_size,
                          int block_size) {
    
    size_t image_size = width * height * channels * sizeof(uint8_t);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    
    uint8_t *d_input, *d_output;
    float *d_kern;
    
    CUDA_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_CHECK(cudaMalloc(&d_output, image_size));
    CUDA_CHECK(cudaMalloc(&d_kern, kernel_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, input, image_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kern, kernel, kernel_bytes, cudaMemcpyHostToDevice));
    
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size,
              (height + block_size - 1) / block_size);
    
    convolve_kernel_global<<<grid, block>>>(d_input, d_output, width, height, 
                                            channels, d_kern, kernel_size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kern));
}

void convolve_cuda_constant(const uint8_t* input, uint8_t* output,
                            int width, int height, int channels,
                            const float* kernel, int kernel_size,
                            int block_size) {
    
    size_t image_size = width * height * channels * sizeof(uint8_t);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    
    uint8_t *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_CHECK(cudaMalloc(&d_output, image_size));
    CUDA_CHECK(cudaMemcpy(d_input, input, image_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, kernel, kernel_bytes));
    
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size,
              (height + block_size - 1) / block_size);
    
    convolve_kernel_constant<<<grid, block>>>(d_input, d_output, width, height, 
                                              channels, kernel_size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void convolve_cuda_shared(const uint8_t* input, uint8_t* output,
                          int width, int height, int channels,
                          const float* kernel, int kernel_size,
                          int block_size) {
    
    size_t image_size = width * height * channels * sizeof(uint8_t);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    
    uint8_t *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_CHECK(cudaMalloc(&d_output, image_size));
    CUDA_CHECK(cudaMemcpy(d_input, input, image_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, kernel, kernel_bytes));
    
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size,
              (height + block_size - 1) / block_size);
    
    int kernel_radius = kernel_size / 2;
    
    if (block_size == 16 && kernel_radius == 1) {
        convolve_kernel_shared<16, 1><<<grid, block>>>(d_input, d_output, width, height, channels, kernel_size);
    } else if (block_size == 16 && kernel_radius == 2) {
        convolve_kernel_shared<16, 2><<<grid, block>>>(d_input, d_output, width, height, channels, kernel_size);
    } else if (block_size == 16 && kernel_radius == 3) {
        convolve_kernel_shared<16, 3><<<grid, block>>>(d_input, d_output, width, height, channels, kernel_size);
    } else {
        convolve_kernel_constant<<<grid, block>>>(d_input, d_output, width, height, channels, kernel_size);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void convolve_cuda_optimized(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             const float* kernel, int kernel_size,
                             int block_size) {
    convolve_cuda_shared(input, output, width, height, channels, kernel, kernel_size, block_size);
}

// ============================================================================
// High-level Image API
// ============================================================================

Image convolve_cuda(const Image& input, const ConvKernel& kernel,
                    CudaOptLevel opt_level, int block_size) {
    
    Image output(input.width, input.height, input.channels);
    
    switch (opt_level) {
        case CudaOptLevel::GLOBAL:
            convolve_cuda_global(input.data.data(), output.data.data(),
                                input.width, input.height, input.channels,
                                kernel.data.data(), kernel.size, block_size);
            break;
        case CudaOptLevel::CONSTANT:
            convolve_cuda_constant(input.data.data(), output.data.data(),
                                  input.width, input.height, input.channels,
                                  kernel.data.data(), kernel.size, block_size);
            break;
        default:
            convolve_cuda_shared(input.data.data(), output.data.data(),
                                input.width, input.height, input.channels,
                                kernel.data.data(), kernel.size, block_size);
    }
    
    return output;
}

Image convolve_cuda_timed(const Image& input, const ConvKernel& kernel,
                          double& time_ms,
                          CudaOptLevel opt_level, int block_size) {
    
    Image output(input.width, input.height, input.channels);
    
    size_t image_size = input.width * input.height * input.channels;
    size_t kernel_bytes = kernel.size * kernel.size * sizeof(float);
    
    uint8_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_CHECK(cudaMalloc(&d_output, image_size));
    CUDA_CHECK(cudaMemcpy(d_input, input.data.data(), image_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, kernel.data.data(), kernel_bytes));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    dim3 block(block_size, block_size);
    dim3 grid((input.width + block_size - 1) / block_size,
              (input.height + block_size - 1) / block_size);
    
    CUDA_CHECK(cudaEventRecord(start));
    
    int kernel_radius = kernel.size / 2;
    if (block_size == 16 && kernel_radius == 1) {
        convolve_kernel_shared<16, 1><<<grid, block>>>(d_input, d_output, 
            input.width, input.height, input.channels, kernel.size);
    } else {
        convolve_kernel_constant<<<grid, block>>>(d_input, d_output,
            input.width, input.height, input.channels, kernel.size);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    time_ms = static_cast<double>(elapsed);
    
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_output, image_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return output;
}

std::map<std::string, double> benchmark_cuda_implementations(
    const Image& input, const ConvKernel& kernel,
    int iterations, int block_size) {
    
    std::map<std::string, double> results;
    double time;
    
    // Warm-up
    convolve_cuda_timed(input, kernel, time, CudaOptLevel::SHARED, block_size);
    
    // Global
    double total = 0;
    for (int i = 0; i < iterations; i++) {
        convolve_cuda(input, kernel, CudaOptLevel::GLOBAL, block_size);
        // Manual timing for global
    }
    results["cuda_global"] = total / iterations;
    
    // Constant
    total = 0;
    for (int i = 0; i < iterations; i++) {
        convolve_cuda_timed(input, kernel, time, CudaOptLevel::CONSTANT, block_size);
        total += time;
    }
    results["cuda_constant"] = total / iterations;
    
    // Shared
    total = 0;
    for (int i = 0; i < iterations; i++) {
        convolve_cuda_timed(input, kernel, time, CudaOptLevel::SHARED, block_size);
        total += time;
    }
    results["cuda_shared"] = total / iterations;
    
    return results;
}

const char* opt_level_to_string(CudaOptLevel level) {
    switch (level) {
        case CudaOptLevel::GLOBAL: return "global";
        case CudaOptLevel::CONSTANT: return "constant";
        case CudaOptLevel::SHARED: return "shared";
        case CudaOptLevel::FULL: return "full";
        default: return "unknown";
    }
}

CudaOptLevel string_to_opt_level(const std::string& str) {
    if (str == "global") return CudaOptLevel::GLOBAL;
    if (str == "constant") return CudaOptLevel::CONSTANT;
    if (str == "shared") return CudaOptLevel::SHARED;
    return CudaOptLevel::FULL;
}
