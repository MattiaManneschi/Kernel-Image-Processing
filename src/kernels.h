#ifndef KERNELS_H
#define KERNELS_H

#include <string>
#include <vector>
#include <map>
#include <stdexcept>





struct ConvKernel {
    std::string name;           
    std::string id;             
    int size;                   
    std::vector<float> data;    
    bool normalize;             
    
    ConvKernel() : size(0), normalize(true) {}
    
    ConvKernel(const std::string& name_, const std::string& id_, int size_,
               const std::vector<float>& data_, bool normalize_ = true)
        : name(name_), id(id_), size(size_), data(data_), normalize(normalize_) {}
    
    
    float at(int x, int y) const {
        return data[y * size + x];
    }
    
    
    int radius() const {
        return size / 2;
    }
    
    
    bool is_valid() const {
        return size > 0 && size % 2 == 1 && 
               data.size() == static_cast<size_t>(size * size);
    }
    
    
    void print() const;
};





namespace Kernels {


ConvKernel box_blur_3x3();
ConvKernel box_blur_5x5();
ConvKernel box_blur_7x7();


ConvKernel gaussian_3x3();
ConvKernel gaussian_5x5();
ConvKernel gaussian_7x7();


ConvKernel sharpen_3x3();
ConvKernel sharpen_strong_3x3();


ConvKernel sobel_x_3x3();
ConvKernel sobel_y_3x3();


ConvKernel prewitt_x_3x3();
ConvKernel prewitt_y_3x3();


ConvKernel laplacian_3x3();
ConvKernel laplacian_diagonal_3x3();


ConvKernel emboss_3x3();


ConvKernel identity_3x3();

} 





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

#endif 
