#ifndef KERNELS_H
#define KERNELS_H

#include <string>
#include <vector>
#include <map>
#include <stdexcept>

// ============================================================================
// Convolution Kernel structure
// ============================================================================

struct ConvKernel {
    std::string name;           // Human-readable name
    std::string id;             // Identifier for CLI
    int size;                   // Kernel size (3, 5, 7, etc.)
    std::vector<float> data;    // Kernel coefficients (row-major)
    bool normalize;             // Whether kernel is normalized (sum = 1)
    
    ConvKernel() : size(0), normalize(true) {}
    
    ConvKernel(const std::string& name_, const std::string& id_, int size_,
               const std::vector<float>& data_, bool normalize_ = true)
        : name(name_), id(id_), size(size_), data(data_), normalize(normalize_) {}
    
    // Get kernel value at position (x, y)
    float at(int x, int y) const {
        return data[y * size + x];
    }
    
    // Get kernel radius
    int radius() const {
        return size / 2;
    }
    
    // Check if kernel is valid
    bool is_valid() const {
        return size > 0 && size % 2 == 1 && 
               data.size() == static_cast<size_t>(size * size);
    }
    
    // Print kernel to console
    void print() const;
};

// ============================================================================
// Predefined kernels
// ============================================================================

namespace Kernels {

// Box blur (averaging)
ConvKernel box_blur_3x3();
ConvKernel box_blur_5x5();
ConvKernel box_blur_7x7();

// Gaussian blur
ConvKernel gaussian_3x3();
ConvKernel gaussian_5x5();
ConvKernel gaussian_7x7();

// Sharpening
ConvKernel sharpen_3x3();
ConvKernel sharpen_strong_3x3();

// Edge detection - Sobel
ConvKernel sobel_x_3x3();
ConvKernel sobel_y_3x3();

// Edge detection - Prewitt
ConvKernel prewitt_x_3x3();
ConvKernel prewitt_y_3x3();

// Edge detection - Laplacian
ConvKernel laplacian_3x3();
ConvKernel laplacian_diagonal_3x3();

// Emboss
ConvKernel emboss_3x3();

// Identity (no change)
ConvKernel identity_3x3();

} // namespace Kernels

// ============================================================================
// Kernel factory
// ============================================================================

/**
 * Get kernel by name and size
 * 
 * @param name Kernel type: "box", "gaussian", "sharpen", "sobel_x", "sobel_y", etc.
 * @param size Kernel size (3, 5, or 7)
 * @return Requested kernel
 * @throws std::runtime_error if kernel not found
 */
ConvKernel get_kernel(const std::string& name, int size = 3);

/**
 * Get list of available kernel names
 */
std::vector<std::string> get_available_kernels();

/**
 * Create custom Gaussian kernel
 * 
 * @param size Kernel size (must be odd)
 * @param sigma Standard deviation
 * @return Custom Gaussian kernel
 */
ConvKernel create_gaussian_kernel(int size, float sigma);

/**
 * Create custom box blur kernel
 * 
 * @param size Kernel size (must be odd)
 * @return Custom box blur kernel
 */
ConvKernel create_box_kernel(int size);

#endif // KERNELS_H
