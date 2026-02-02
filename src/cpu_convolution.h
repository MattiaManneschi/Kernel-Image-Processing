#ifndef CPU_CONVOLUTION_H
#define CPU_CONVOLUTION_H

#include <cstdint>
#include "image_io.h"
#include "kernels.h"





/**
 * Apply convolution filter to image (sequential CPU version)
 * 
 * @param input Input pixel data
 * @param output Output pixel data (must be pre-allocated)
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param kernel Convolution kernel coefficients
 * @param kernel_size Kernel size (e.g., 3 for 3x3)
 */
void convolve_cpu(const uint8_t* input, uint8_t* output,
                  int width, int height, int channels,
                  const float* kernel, int kernel_size);

/**
 * Apply convolution filter to Image object
 * 
 * @param input Input image
 * @param kernel Convolution kernel
 * @return Filtered image
 */
Image convolve_cpu(const Image& input, const ConvKernel& kernel);

/**
 * Apply convolution with timing
 * 
 * @param input Input image
 * @param kernel Convolution kernel
 * @param time_ms Output parameter for elapsed time in milliseconds
 * @return Filtered image
 */
Image convolve_cpu_timed(const Image& input, const ConvKernel& kernel, double& time_ms);





enum class EdgeMode {
    CLAMP,      
    ZERO,       
    MIRROR,     
    WRAP        
};

/**
 * Apply convolution with configurable edge handling
 * 
 * @param input Input pixel data
 * @param output Output pixel data
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param kernel Convolution kernel coefficients
 * @param kernel_size Kernel size
 * @param edge_mode How to handle image edges
 */
void convolve_cpu_edge(const uint8_t* input, uint8_t* output,
                       int width, int height, int channels,
                       const float* kernel, int kernel_size,
                       EdgeMode edge_mode);





/**
 * Check if kernel is separable (can be decomposed into horizontal and vertical passes)
 * 
 * @param kernel Convolution kernel
 * @param h_kernel Output horizontal 1D kernel
 * @param v_kernel Output vertical 1D kernel
 * @return true if kernel is separable
 */
bool is_separable(const ConvKernel& kernel, 
                  std::vector<float>& h_kernel, 
                  std::vector<float>& v_kernel);

/**
 * Apply separable convolution (horizontal then vertical pass)
 * More efficient for separable kernels: O(n) instead of O(n²) per pixel
 * 
 * @param input Input image
 * @param h_kernel Horizontal 1D kernel
 * @param v_kernel Vertical 1D kernel
 * @return Filtered image
 */
Image convolve_cpu_separable(const Image& input,
                             const std::vector<float>& h_kernel,
                             const std::vector<float>& v_kernel);

#endif 
