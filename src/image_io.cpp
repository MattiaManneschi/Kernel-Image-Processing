#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image_io.h"
#include "utils.h"
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <cmath>
#include <random>
#include <algorithm>

// ============================================================================
// Image loading
// ============================================================================

Image load_image(const std::string& path, int force_channels) {
    Image image;
    
    int width, height, channels;
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, force_channels);
    
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path + 
                                 " - " + stbi_failure_reason());
    }
    
    image.width = width;
    image.height = height;
    image.channels = force_channels > 0 ? force_channels : channels;
    
    size_t size = width * height * image.channels;
    image.data.assign(data, data + size);
    
    stbi_image_free(data);
    
    LOG_DEBUG("Loaded image: ", path, " (", width, "x", height, "x", image.channels, ")");
    
    return image;
}

// ============================================================================
// Image saving
// ============================================================================

void save_image(const std::string& path, const Image& image, int quality) {
    if (!image.is_valid()) {
        throw std::runtime_error("Cannot save invalid image");
    }
    
    std::string ext = get_extension(path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    int success = 0;
    
    if (ext == "png") {
        success = stbi_write_png(path.c_str(), image.width, image.height, 
                                  image.channels, image.data.data(), 
                                  image.width * image.channels);
    } else if (ext == "jpg" || ext == "jpeg") {
        success = stbi_write_jpg(path.c_str(), image.width, image.height,
                                  image.channels, image.data.data(), quality);
    } else if (ext == "bmp") {
        success = stbi_write_bmp(path.c_str(), image.width, image.height,
                                  image.channels, image.data.data());
    } else if (ext == "tga") {
        success = stbi_write_tga(path.c_str(), image.width, image.height,
                                  image.channels, image.data.data());
    } else {
        throw std::runtime_error("Unsupported image format: " + ext);
    }
    
    if (!success) {
        throw std::runtime_error("Failed to save image: " + path);
    }
    
    LOG_DEBUG("Saved image: ", path);
}

// ============================================================================
// Test image generation
// ============================================================================

Image create_test_image(int width, int height, int channels, const std::string& pattern) {
    Image image(width, height, channels);
    
    if (pattern == "gradient") {
        // Diagonal gradient
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float fx = static_cast<float>(x) / (width - 1);
                float fy = static_cast<float>(y) / (height - 1);
                
                for (int c = 0; c < channels; c++) {
                    float value;
                    if (c == 0) value = fx * 255;           // R: horizontal gradient
                    else if (c == 1) value = fy * 255;      // G: vertical gradient
                    else if (c == 2) value = (fx + fy) * 127.5f; // B: diagonal
                    else value = 255;                        // A: opaque
                    
                    image.at(x, y, c) = static_cast<uint8_t>(clamp(value, 0.0f, 255.0f));
                }
            }
        }
    } else if (pattern == "checkerboard") {
        // Checkerboard pattern
        int block_size = std::max(width, height) / 8;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                bool white = ((x / block_size) + (y / block_size)) % 2 == 0;
                uint8_t value = white ? 255 : 0;
                
                for (int c = 0; c < channels; c++) {
                    if (c < 3) image.at(x, y, c) = value;
                    else image.at(x, y, c) = 255;  // Alpha
                }
            }
        }
    } else if (pattern == "noise") {
        // Random noise
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    if (c < 3) image.at(x, y, c) = static_cast<uint8_t>(dis(gen));
                    else image.at(x, y, c) = 255;  // Alpha
                }
            }
        }
    } else if (pattern == "solid") {
        // Solid gray
        image.fill(128);
        if (channels == 4) {
            // Set alpha to 255
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    image.at(x, y, 3) = 255;
                }
            }
        }
    } else if (pattern == "circles") {
        // Concentric circles (good for testing blur)
        image.fill(255);
        int cx = width / 2;
        int cy = height / 2;
        int max_radius = std::min(width, height) / 2;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = std::sqrt(dx*dx + dy*dy);
                
                // Create rings
                int ring = static_cast<int>(dist / (max_radius / 10.0f));
                uint8_t value = (ring % 2 == 0) ? 0 : 255;
                
                for (int c = 0; c < std::min(channels, 3); c++) {
                    image.at(x, y, c) = value;
                }
                if (channels == 4) image.at(x, y, 3) = 255;
            }
        }
    } else {
        throw std::runtime_error("Unknown pattern: " + pattern);
    }
    
    return image;
}

// ============================================================================
// Image conversion
// ============================================================================

Image to_grayscale(const Image& image) {
    if (image.channels == 1) {
        return image.clone();
    }
    
    Image gray(image.width, image.height, 1);
    
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            // Standard luminosity formula
            float r = image.at(x, y, 0);
            float g = image.at(x, y, 1);
            float b = image.at(x, y, 2);
            float lum = 0.299f * r + 0.587f * g + 0.114f * b;
            gray.at(x, y, 0) = static_cast<uint8_t>(clamp(lum, 0.0f, 255.0f));
        }
    }
    
    return gray;
}

// ============================================================================
// Image comparison
// ============================================================================

bool images_equal(const Image& img1, const Image& img2, int tolerance) {
    if (img1.width != img2.width || img1.height != img2.height || 
        img1.channels != img2.channels) {
        return false;
    }
    
    for (size_t i = 0; i < img1.data.size(); i++) {
        int diff = std::abs(static_cast<int>(img1.data[i]) - static_cast<int>(img2.data[i]));
        if (diff > tolerance) {
            return false;
        }
    }
    
    return true;
}

double compute_psnr(const Image& img1, const Image& img2) {
    if (img1.width != img2.width || img1.height != img2.height || 
        img1.channels != img2.channels) {
        throw std::runtime_error("Images must have same dimensions for PSNR");
    }
    
    double mse = 0.0;
    for (size_t i = 0; i < img1.data.size(); i++) {
        double diff = static_cast<double>(img1.data[i]) - static_cast<double>(img2.data[i]);
        mse += diff * diff;
    }
    mse /= img1.data.size();
    
    if (mse < 1e-10) {
        return 100.0;  // Identical images
    }
    
    double max_val = 255.0;
    return 10.0 * std::log10((max_val * max_val) / mse);
}

// ============================================================================
// Utility functions
// ============================================================================

void print_image_info(const Image& image, const std::string& name) {
    std::cout << name << ": " 
              << image.width << "x" << image.height 
              << " (" << image.channels << " channels, "
              << image.size_bytes() << " bytes)" << std::endl;
}
