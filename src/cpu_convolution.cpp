#include "cpu_convolution.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <chrono>

// ============================================================================
// Helper function: get pixel with edge handling
// ============================================================================

static inline int clamp_coord(int val, int min_val, int max_val) {
    return std::max(min_val, std::min(val, max_val));
}

static inline int mirror_coord(int val, int size) {
    if (val < 0) return -val - 1;
    if (val >= size) return 2 * size - val - 1;
    return val;
}

static inline int wrap_coord(int val, int size) {
    return ((val % size) + size) % size;
}

// ============================================================================
// Main convolution implementation
// ============================================================================

void convolve_cpu(const uint8_t* input, uint8_t* output,
                  int width, int height, int channels,
                  const float* kernel, int kernel_size) {
    
    const int half = kernel_size / 2;
    
    // Process each pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Process each channel
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                // Apply kernel
                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        // Clamp coordinates to image bounds
                        int px = clamp_coord(x + kx, 0, width - 1);
                        int py = clamp_coord(y + ky, 0, height - 1);
                        
                        // Get pixel value
                        int img_idx = (py * width + px) * channels + c;
                        
                        // Get kernel value
                        int k_idx = (ky + half) * kernel_size + (kx + half);
                        
                        sum += static_cast<float>(input[img_idx]) * kernel[k_idx];
                    }
                }
                
                // Clamp result to valid range and store
                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, sum))
                );
            }
        }
    }
}

// ============================================================================
// Convolution with edge handling options
// ============================================================================

void convolve_cpu_edge(const uint8_t* input, uint8_t* output,
                       int width, int height, int channels,
                       const float* kernel, int kernel_size,
                       EdgeMode edge_mode) {
    
    const int half = kernel_size / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        
                        // Handle edges based on mode
                        switch (edge_mode) {
                            case EdgeMode::CLAMP:
                                px = clamp_coord(px, 0, width - 1);
                                py = clamp_coord(py, 0, height - 1);
                                break;
                            case EdgeMode::ZERO:
                                if (px < 0 || px >= width || py < 0 || py >= height) {
                                    continue;  // Skip, adds 0
                                }
                                break;
                            case EdgeMode::MIRROR:
                                px = mirror_coord(px, width);
                                py = mirror_coord(py, height);
                                break;
                            case EdgeMode::WRAP:
                                px = wrap_coord(px, width);
                                py = wrap_coord(py, height);
                                break;
                        }
                        
                        int img_idx = (py * width + px) * channels + c;
                        int k_idx = (ky + half) * kernel_size + (kx + half);
                        
                        sum += static_cast<float>(input[img_idx]) * kernel[k_idx];
                    }
                }
                
                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, sum))
                );
            }
        }
    }
}

// ============================================================================
// Image-based wrappers
// ============================================================================

Image convolve_cpu(const Image& input, const ConvKernel& kernel) {
    if (!input.is_valid()) {
        throw std::runtime_error("Invalid input image");
    }
    if (!kernel.is_valid()) {
        throw std::runtime_error("Invalid convolution kernel");
    }
    
    Image output(input.width, input.height, input.channels);
    
    convolve_cpu(input.data.data(), output.data.data(),
                 input.width, input.height, input.channels,
                 kernel.data.data(), kernel.size);
    
    return output;
}

Image convolve_cpu_timed(const Image& input, const ConvKernel& kernel, double& time_ms) {
    if (!input.is_valid()) {
        throw std::runtime_error("Invalid input image");
    }
    if (!kernel.is_valid()) {
        throw std::runtime_error("Invalid convolution kernel");
    }
    
    Image output(input.width, input.height, input.channels);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    convolve_cpu(input.data.data(), output.data.data(),
                 input.width, input.height, input.channels,
                 kernel.data.data(), kernel.size);
    
    auto end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return output;
}

// ============================================================================
// Separable convolution
// ============================================================================

bool is_separable(const ConvKernel& kernel, 
                  std::vector<float>& h_kernel, 
                  std::vector<float>& v_kernel) {
    // Simple check for common separable kernels
    // A proper implementation would use SVD
    
    int size = kernel.size;
    h_kernel.resize(size);
    v_kernel.resize(size);
    
    // Check if kernel can be expressed as outer product of two vectors
    // For simplicity, we check if all rows are proportional
    
    // Get first column as vertical kernel
    for (int y = 0; y < size; y++) {
        v_kernel[y] = kernel.at(0, y);
    }
    
    // Get first row as horizontal kernel  
    for (int x = 0; x < size; x++) {
        h_kernel[x] = kernel.at(x, 0);
    }
    
    // Normalize
    float v_sum = 0, h_sum = 0;
    for (int i = 0; i < size; i++) {
        v_sum += std::abs(v_kernel[i]);
        h_sum += std::abs(h_kernel[i]);
    }
    
    if (v_sum < 1e-6 || h_sum < 1e-6) {
        return false;
    }
    
    // Check if outer product matches original kernel
    float tolerance = 1e-4f;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float expected = (v_kernel[y] / v_kernel[0]) * h_kernel[x];
            float actual = kernel.at(x, y);
            if (std::abs(expected - actual) > tolerance * std::abs(actual + 1e-6)) {
                return false;
            }
        }
    }
    
    // Adjust kernels so their outer product gives the original
    float scale = std::sqrt(std::abs(kernel.at(0, 0)) / (h_kernel[0] * v_kernel[0] + 1e-10f));
    for (int i = 0; i < size; i++) {
        h_kernel[i] *= scale;
        v_kernel[i] *= scale;
    }
    
    return true;
}

Image convolve_cpu_separable(const Image& input,
                             const std::vector<float>& h_kernel,
                             const std::vector<float>& v_kernel) {
    
    int size = static_cast<int>(h_kernel.size());
    int half = size / 2;
    
    // Temporary buffer for horizontal pass
    Image temp(input.width, input.height, input.channels);
    
    // Horizontal pass
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            for (int c = 0; c < input.channels; c++) {
                float sum = 0.0f;
                for (int k = -half; k <= half; k++) {
                    int px = clamp_coord(x + k, 0, input.width - 1);
                    sum += static_cast<float>(input.at(px, y, c)) * h_kernel[k + half];
                }
                temp.at(x, y, c) = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, sum))
                );
            }
        }
    }
    
    // Vertical pass
    Image output(input.width, input.height, input.channels);
    
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            for (int c = 0; c < input.channels; c++) {
                float sum = 0.0f;
                for (int k = -half; k <= half; k++) {
                    int py = clamp_coord(y + k, 0, input.height - 1);
                    sum += static_cast<float>(temp.at(x, py, c)) * v_kernel[k + half];
                }
                output.at(x, y, c) = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, sum))
                );
            }
        }
    }
    
    return output;
}
