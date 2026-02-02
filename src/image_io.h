#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>





struct Image {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;
    
    Image() : width(0), height(0), channels(0) {}
    
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }
    
    
    uint8_t& at(int x, int y, int c) {
        return data[(y * width + x) * channels + c];
    }
    
    const uint8_t& at(int x, int y, int c) const {
        return data[(y * width + x) * channels + c];
    }
    
    
    size_t size_bytes() const {
        return data.size();
    }
    
    
    size_t num_pixels() const {
        return width * height;
    }
    
    
    bool is_valid() const {
        return width > 0 && height > 0 && channels > 0 && !data.empty();
    }
    
    
    void clear() {
        width = height = channels = 0;
        data.clear();
    }
    
    
    void fill(uint8_t value) {
        std::fill(data.begin(), data.end(), value);
    }
    
    
    Image clone() const {
        Image copy;
        copy.width = width;
        copy.height = height;
        copy.channels = channels;
        copy.data = data;
        return copy;
    }
};





/**
 * Load image from file
 * Supports: PNG, JPG, BMP, TGA, PSD, GIF, HDR, PIC, PNM
 * 
 * @param path Path to image file
 * @param force_channels Force number of channels (0 = auto)
 * @return Loaded image
 * @throws std::runtime_error if loading fails
 */
Image load_image(const std::string& path, int force_channels = 0);

/**
 * Save image to file
 * Format determined by extension: .png, .jpg, .bmp, .tga
 * 
 * @param path Output path
 * @param image Image to save
 * @param quality JPEG quality (1-100), ignored for other formats
 * @throws std::runtime_error if saving fails
 */
void save_image(const std::string& path, const Image& image, int quality = 95);

/**
 * Create synthetic test image
 * 
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (1, 3, or 4)
 * @param pattern Pattern type: "gradient", "checkerboard", "noise", "solid"
 * @return Generated image
 */
Image create_test_image(int width, int height, int channels = 3, 
                        const std::string& pattern = "gradient");

/**
 * Convert image to grayscale
 * 
 * @param image Input image (RGB or RGBA)
 * @return Grayscale image (1 channel)
 */
Image to_grayscale(const Image& image);

/**
 * Compare two images for equality (with tolerance)
 * 
 * @param img1 First image
 * @param img2 Second image  
 * @param tolerance Maximum allowed difference per pixel
 * @return true if images match within tolerance
 */
bool images_equal(const Image& img1, const Image& img2, int tolerance = 0);

/**
 * Compute PSNR between two images
 * 
 * @param img1 First image
 * @param img2 Second image
 * @return PSNR in dB (higher = more similar)
 */
double compute_psnr(const Image& img1, const Image& img2);

/**
 * Print image info
 */
void print_image_info(const Image& image, const std::string& name = "Image");

#endif 
