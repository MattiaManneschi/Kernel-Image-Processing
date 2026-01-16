#ifndef GPU_CONVOLUTION_CUH
#define GPU_CONVOLUTION_CUH

#include <cstdint>
#include "image_io.h"
#include "kernels.h"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CUDA Device Info
// ============================================================================

/**
 * Print CUDA device information
 */
void print_cuda_info();

/**
 * Check if CUDA is available
 * @return true if CUDA device is available
 */
bool cuda_available();

/**
 * Get number of CUDA devices
 */
int get_cuda_device_count();

// ============================================================================
// GPU Convolution - Different Implementations
// ============================================================================

/**
 * Optimization levels for CUDA convolution
 */
enum class CudaOptLevel {
    GLOBAL,         // Basic global memory implementation
    CONSTANT,       // Kernel in constant memory
    SHARED,         // Shared memory tiling
    FULL            // All optimizations
};

/**
 * Apply convolution using CUDA (basic global memory version)
 * 
 * @param input Input pixel data (host)
 * @param output Output pixel data (host, pre-allocated)
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param kernel Convolution kernel (host)
 * @param kernel_size Kernel size
 * @param block_size CUDA block size (default 16)
 */
void convolve_cuda_global(const uint8_t* input, uint8_t* output,
                          int width, int height, int channels,
                          const float* kernel, int kernel_size,
                          int block_size = 16);

/**
 * Apply convolution using CUDA with constant memory for kernel
 */
void convolve_cuda_constant(const uint8_t* input, uint8_t* output,
                            int width, int height, int channels,
                            const float* kernel, int kernel_size,
                            int block_size = 16);

/**
 * Apply convolution using CUDA with shared memory tiling
 */
void convolve_cuda_shared(const uint8_t* input, uint8_t* output,
                          int width, int height, int channels,
                          const float* kernel, int kernel_size,
                          int block_size = 16);

/**
 * Apply convolution with all optimizations
 */
void convolve_cuda_optimized(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             const float* kernel, int kernel_size,
                             int block_size = 16);

// ============================================================================
// High-level Image API
// ============================================================================

/**
 * Apply CUDA convolution to Image object
 * 
 * @param input Input image
 * @param kernel Convolution kernel
 * @param opt_level Optimization level
 * @param block_size CUDA block size
 * @return Filtered image
 */
Image convolve_cuda(const Image& input, const ConvKernel& kernel,
                    CudaOptLevel opt_level = CudaOptLevel::FULL,
                    int block_size = 16);

/**
 * Apply CUDA convolution with timing
 * 
 * @param input Input image
 * @param kernel Convolution kernel
 * @param time_ms Output: elapsed time in milliseconds
 * @param opt_level Optimization level
 * @param block_size CUDA block size
 * @return Filtered image
 */
Image convolve_cuda_timed(const Image& input, const ConvKernel& kernel,
                          double& time_ms,
                          CudaOptLevel opt_level = CudaOptLevel::FULL,
                          int block_size = 16);

/**
 * Benchmark different CUDA implementations
 * 
 * @param input Input image
 * @param kernel Convolution kernel
 * @param iterations Number of iterations for timing
 * @param block_size CUDA block size
 * @return Map of implementation name to average time in ms
 */
#include <map>
#include <string>
std::map<std::string, double> benchmark_cuda_implementations(
    const Image& input, const ConvKernel& kernel,
    int iterations = 10, int block_size = 16);

// ============================================================================
// Utility functions
// ============================================================================

/**
 * Convert optimization level to string
 */
const char* opt_level_to_string(CudaOptLevel level);

/**
 * Convert string to optimization level
 */
CudaOptLevel string_to_opt_level(const std::string& str);

#endif // GPU_CONVOLUTION_CUH
