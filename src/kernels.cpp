#include "kernels.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>





void ConvKernel::print() const {
    std::cout << name << " (" << size << "x" << size << "):" << std::endl;
    for (int y = 0; y < size; y++) {
        std::cout << "  [";
        for (int x = 0; x < size; x++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << at(x, y);
            if (x < size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}





namespace Kernels {

ConvKernel box_blur_3x3() {
    float v = 1.0f / 9.0f;
    return ConvKernel("Box Blur 3x3", "box", 3, {
        v, v, v,
        v, v, v,
        v, v, v
    });
}

ConvKernel box_blur_5x5() {
    float v = 1.0f / 25.0f;
    return ConvKernel("Box Blur 5x5", "box", 5, {
        v, v, v, v, v,
        v, v, v, v, v,
        v, v, v, v, v,
        v, v, v, v, v,
        v, v, v, v, v
    });
}

ConvKernel box_blur_7x7() {
    float v = 1.0f / 49.0f;
    std::vector<float> data(49, v);
    return ConvKernel("Box Blur 7x7", "box", 7, data);
}





ConvKernel gaussian_3x3() {
    float s = 16.0f;
    return ConvKernel("Gaussian Blur 3x3", "gaussian", 3, {
        1/s, 2/s, 1/s,
        2/s, 4/s, 2/s,
        1/s, 2/s, 1/s
    });
}

ConvKernel gaussian_5x5() {
    float s = 256.0f;
    return ConvKernel("Gaussian Blur 5x5", "gaussian", 5, {
        1/s,  4/s,  6/s,  4/s, 1/s,
        4/s, 16/s, 24/s, 16/s, 4/s,
        6/s, 24/s, 36/s, 24/s, 6/s,
        4/s, 16/s, 24/s, 16/s, 4/s,
        1/s,  4/s,  6/s,  4/s, 1/s
    });
}

ConvKernel gaussian_7x7() {
    
    return create_gaussian_kernel(7, 1.0f);
}





ConvKernel sharpen_3x3() {
    return ConvKernel("Sharpen 3x3", "sharpen", 3, {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    }, false);
}

ConvKernel sharpen_strong_3x3() {
    return ConvKernel("Sharpen Strong 3x3", "sharpen_strong", 3, {
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    }, false);
}





ConvKernel sobel_x_3x3() {
    return ConvKernel("Sobel X 3x3", "sobel_x", 3, {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    }, false);
}

ConvKernel sobel_y_3x3() {
    return ConvKernel("Sobel Y 3x3", "sobel_y", 3, {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    }, false);
}





ConvKernel prewitt_x_3x3() {
    return ConvKernel("Prewitt X 3x3", "prewitt_x", 3, {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    }, false);
}

ConvKernel prewitt_y_3x3() {
    return ConvKernel("Prewitt Y 3x3", "prewitt_y", 3, {
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    }, false);
}





ConvKernel laplacian_3x3() {
    return ConvKernel("Laplacian 3x3", "laplacian", 3, {
        0,  1, 0,
        1, -4, 1,
        0,  1, 0
    }, false);
}

ConvKernel laplacian_diagonal_3x3() {
    return ConvKernel("Laplacian Diagonal 3x3", "laplacian_diag", 3, {
        1,  1, 1,
        1, -8, 1,
        1,  1, 1
    }, false);
}





ConvKernel emboss_3x3() {
    return ConvKernel("Emboss 3x3", "emboss", 3, {
        -2, -1, 0,
        -1,  1, 1,
         0,  1, 2
    }, false);
}





ConvKernel identity_3x3() {
    return ConvKernel("Identity 3x3", "identity", 3, {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    }, false);
}

} 





ConvKernel get_kernel(const std::string& name, int size) {
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    
    if (lower_name == "box" || lower_name == "box_blur") {
        if (size == 3) return Kernels::box_blur_3x3();
        if (size == 5) return Kernels::box_blur_5x5();
        if (size == 7) return Kernels::box_blur_7x7();
        return create_box_kernel(size);
    }
    
    
    if (lower_name == "gaussian" || lower_name == "gaussian_blur") {
        if (size == 3) return Kernels::gaussian_3x3();
        if (size == 5) return Kernels::gaussian_5x5();
        if (size == 7) return Kernels::gaussian_7x7();
        return create_gaussian_kernel(size, size / 6.0f);
    }
    
    
    if (lower_name == "sharpen") {
        return Kernels::sharpen_3x3();
    }
    if (lower_name == "sharpen_strong") {
        return Kernels::sharpen_strong_3x3();
    }
    
    
    if (lower_name == "sobel_x" || lower_name == "sobelx") {
        return Kernels::sobel_x_3x3();
    }
    if (lower_name == "sobel_y" || lower_name == "sobely") {
        return Kernels::sobel_y_3x3();
    }
    
    
    if (lower_name == "prewitt_x" || lower_name == "prewittx") {
        return Kernels::prewitt_x_3x3();
    }
    if (lower_name == "prewitt_y" || lower_name == "prewitty") {
        return Kernels::prewitt_y_3x3();
    }
    
    
    if (lower_name == "laplacian") {
        return Kernels::laplacian_3x3();
    }
    if (lower_name == "laplacian_diag") {
        return Kernels::laplacian_diagonal_3x3();
    }
    
    
    if (lower_name == "emboss") {
        return Kernels::emboss_3x3();
    }
    
    
    if (lower_name == "identity" || lower_name == "none") {
        return Kernels::identity_3x3();
    }
    
    throw std::runtime_error("Unknown kernel: " + name);
}

std::vector<std::string> get_available_kernels() {
    return {
        "box", "gaussian", "sharpen", "sharpen_strong",
        "sobel_x", "sobel_y", "prewitt_x", "prewitt_y",
        "laplacian", "laplacian_diag", "emboss", "identity"
    };
}





ConvKernel create_gaussian_kernel(int size, float sigma) {
    if (size % 2 == 0) {
        throw std::runtime_error("Kernel size must be odd");
    }
    
    std::vector<float> data(size * size);
    int half = size / 2;
    float sum = 0.0f;
    
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float value = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            data[(y + half) * size + (x + half)] = value;
            sum += value;
        }
    }
    
    
    for (float& v : data) {
        v /= sum;
    }
    
    std::string name = "Gaussian " + std::to_string(size) + "x" + std::to_string(size) + 
                       " (σ=" + std::to_string(sigma) + ")";
    return ConvKernel(name, "gaussian", size, data);
}

ConvKernel create_box_kernel(int size) {
    if (size % 2 == 0) {
        throw std::runtime_error("Kernel size must be odd");
    }
    
    float value = 1.0f / (size * size);
    std::vector<float> data(size * size, value);
    
    std::string name = "Box Blur " + std::to_string(size) + "x" + std::to_string(size);
    return ConvKernel(name, "box", size, data);
}
