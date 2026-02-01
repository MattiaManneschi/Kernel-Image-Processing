#include "advanced_tests.h"
#include "cpu_convolution.h"
#include "gpu_convolution.cuh"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <random>

// ============================================================================
// Helper Functions - Image Loading
// ============================================================================

// Find images in a directory
static std::vector<std::string> find_images_in_dir(const std::string& dir) {
    std::vector<std::string> paths;

    // Try common Kodak naming
    for (int i = 1; i <= 24; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/kodim%02d.png", dir.c_str(), i);
        std::ifstream f(filename);
        if (f.good()) {
            paths.push_back(filename);
        }
    }

    return paths;
}

// Load a random image from directory, or return empty if not found
static Image load_random_image(const std::string& dir) {
    auto paths = find_images_in_dir(dir);
    if (paths.empty()) {
        return Image();  // Return invalid image
    }

    // Pick random
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, paths.size() - 1);
    int idx = dis(gen);

    std::cout << "Loading: " << paths[idx] << std::endl;
    return load_image(paths[idx]);
}

// Resize image to target dimensions
static Image resize_image(const Image& src, int target_width, int target_height) {
    if (src.width == target_width && src.height == target_height) {
        return src;
    }

    Image resized(target_width, target_height, src.channels);

    float x_ratio = static_cast<float>(src.width) / target_width;
    float y_ratio = static_cast<float>(src.height) / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            int src_x = static_cast<int>(x * x_ratio);
            int src_y = static_cast<int>(y * y_ratio);
            src_x = std::min(src_x, src.width - 1);
            src_y = std::min(src_y, src.height - 1);

            for (int c = 0; c < src.channels; c++) {
                resized.at(x, y, c) = src.at(src_x, src_y, c);
            }
        }
    }
    return resized;
}

// Global base image for tests (loaded once)
static Image g_base_image;
static bool g_using_real_images = false;

// Forward declarations
static Image create_gradient_image(int width, int height, int channels);
static Image convert_channels(const Image& src, int target_channels);

// Initialize base image from directory or synthetic
static void init_base_image(const std::string& images_dir) {
    if (!images_dir.empty()) {
        std::cout << "Searching for images in: " << images_dir << std::endl;
        try {
            g_base_image = load_random_image(images_dir);
            if (g_base_image.is_valid()) {
                g_using_real_images = true;
                std::cout << ">>> Using REAL image: " << g_base_image.width << "x"
                          << g_base_image.height << ", " << g_base_image.channels << " channels" << std::endl;
                return;
            }
        } catch (const std::exception& e) {
            std::cout << "Failed to load: " << e.what() << std::endl;
        }
        std::cout << "No images found, falling back to synthetic" << std::endl;
    }
    g_using_real_images = false;
    std::cout << ">>> Using SYNTHETIC images" << std::endl;
}

// Get image at target size (from real or synthetic)
static Image get_test_image(int width, int height, int channels = 3) {
    if (g_using_real_images && g_base_image.is_valid()) {
        Image resized = resize_image(g_base_image, width, height);
        // Convert channels if needed
        if (resized.channels != channels) {
            return convert_channels(resized, channels);
        }
        return resized;
    }
    // Fallback to synthetic
    return create_gradient_image(width, height, channels);
}

// ============================================================================
// Helper Functions - Image Generation
// ============================================================================

// Create test images of different types
static Image create_natural_like_image(int width, int height, int channels) {
    // Simulate natural image with varied content
    Image img(width, height, channels);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(128.0f, 40.0f);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                // Add some spatial correlation (smoother than pure noise)
                float base = dist(gen);
                float spatial = 20.0f * std::sin(x * 0.05f) * std::cos(y * 0.03f);
                int val = static_cast<int>(base + spatial);
                img.at(x, y, c) = static_cast<uint8_t>(std::clamp(val, 0, 255));
            }
        }
    }
    return img;
}

static Image create_noise_image(int width, int height, int channels) {
    Image img(width, height, channels);
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(0, 255);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                img.at(x, y, c) = static_cast<uint8_t>(dist(gen));
            }
        }
    }
    return img;
}

static Image create_uniform_image(int width, int height, int channels, uint8_t value = 128) {
    Image img(width, height, channels);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                img.at(x, y, c) = value;
            }
        }
    }
    return img;
}

static Image create_checker_image(int width, int height, int channels, int block_size = 32) {
    Image img(width, height, channels);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t val = ((x / block_size + y / block_size) % 2 == 0) ? 255 : 0;
            for (int c = 0; c < channels; c++) {
                img.at(x, y, c) = val;
            }
        }
    }
    return img;
}

static Image create_gradient_image(int width, int height, int channels) {
    Image img(width, height, channels);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                if (c == 0) img.at(x, y, c) = static_cast<uint8_t>(255 * x / width);
                else if (c == 1) img.at(x, y, c) = static_cast<uint8_t>(255 * y / height);
                else img.at(x, y, c) = static_cast<uint8_t>(128);
            }
        }
    }
    return img;
}

// Convert image to different channel count
static Image convert_channels(const Image& src, int target_channels) {
    if (src.channels == target_channels) return src;

    Image dst(src.width, src.height, target_channels);

    for (int y = 0; y < src.height; y++) {
        for (int x = 0; x < src.width; x++) {
            if (target_channels == 1) {
                // Convert to grayscale
                int gray = 0;
                for (int c = 0; c < src.channels && c < 3; c++) {
                    gray += src.at(x, y, c);
                }
                dst.at(x, y, 0) = static_cast<uint8_t>(gray / std::min(src.channels, 3));
            } else if (target_channels == 3) {
                if (src.channels == 1) {
                    // Grayscale to RGB
                    for (int c = 0; c < 3; c++) {
                        dst.at(x, y, c) = src.at(x, y, 0);
                    }
                } else {
                    // Copy first 3 channels
                    for (int c = 0; c < 3; c++) {
                        dst.at(x, y, c) = src.at(x, y, c % src.channels);
                    }
                }
            } else if (target_channels == 4) {
                // Add alpha channel
                for (int c = 0; c < 3; c++) {
                    dst.at(x, y, c) = src.at(x, y, c % src.channels);
                }
                dst.at(x, y, 3) = 255;  // Full opacity
            }
        }
    }
    return dst;
}

// ============================================================================
// Test 1: Image Types Comparison
// ============================================================================

std::vector<TestResult> test_image_types(const AdvancedTestConfig& config) {
    std::vector<TestResult> results;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 1: Image Types Comparison" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int width = 1024, height = 1024, channels = 3;
    ConvKernel kernel = get_kernel("gaussian", 5);

    // Create different image types
    std::vector<std::pair<std::string, Image>> test_images = {
        {"Natural-like", create_natural_like_image(width, height, channels)},
        {"Random noise", create_noise_image(width, height, channels)},
        {"Uniform gray", create_uniform_image(width, height, channels)},
        {"Checkerboard", create_checker_image(width, height, channels)},
        {"Gradient", create_gradient_image(width, height, channels)}
    };

    std::cout << "\nImage size: " << width << "x" << height << ", Kernel: " << kernel.name << "\n\n";
    std::cout << std::setw(20) << "Image Type"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "CUDA (ms)"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (const auto& [name, image] : test_images) {
        // CPU timing
        std::vector<double> cpu_times;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cpu_timed(image, kernel, t);
            cpu_times.push_back(t);
        }
        double cpu_avg = compute_mean(cpu_times);

        // CUDA timing
        std::vector<double> cuda_times;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cuda_timed(image, kernel, t, CudaOptLevel::SHARED);
            cuda_times.push_back(t);
        }
        double cuda_avg = compute_mean(cuda_times);

        double speedup = cpu_avg / cuda_avg;

        std::cout << std::setw(20) << name
                  << std::setw(15) << std::fixed << std::setprecision(2) << cpu_avg
                  << std::setw(15) << cuda_avg
                  << std::setw(14) << speedup << "x" << std::endl;

        TestResult r;
        r.test_name = "Image Types";
        r.configuration = name;
        r.time_ms = cuda_avg;
        r.speedup = speedup;
        r.throughput = (width * height / 1e6) / (cuda_avg / 1000.0);
        results.push_back(r);
    }

    return results;
}

// ============================================================================
// Test 2: Channel Count Comparison
// ============================================================================

std::vector<TestResult> test_channel_counts(const AdvancedTestConfig& config) {
    std::vector<TestResult> results;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 2: Channel Count Comparison" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int width = 1024, height = 1024;
    ConvKernel kernel = get_kernel("gaussian", 5);

    // Base image (RGB)
    Image base_image = create_gradient_image(width, height, 3);

    std::vector<std::pair<std::string, int>> channel_configs = {
        {"Grayscale (1ch)", 1},
        {"RGB (3ch)", 3},
        {"RGBA (4ch)", 4}
    };

    std::cout << "\nImage size: " << width << "x" << height << ", Kernel: " << kernel.name << "\n\n";
    std::cout << std::setw(20) << "Channels"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "CUDA (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(18) << "Throughput (MP/s)" << std::endl;
    std::cout << std::string(83, '-') << std::endl;

    for (const auto& [name, ch] : channel_configs) {
        Image image = convert_channels(base_image, ch);

        // CPU timing
        std::vector<double> cpu_times;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cpu_timed(image, kernel, t);
            cpu_times.push_back(t);
        }
        double cpu_avg = compute_mean(cpu_times);

        // CUDA timing
        std::vector<double> cuda_times;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cuda_timed(image, kernel, t, CudaOptLevel::SHARED);
            cuda_times.push_back(t);
        }
        double cuda_avg = compute_mean(cuda_times);

        double speedup = cpu_avg / cuda_avg;
        double throughput = (width * height / 1e6) / (cuda_avg / 1000.0);

        std::cout << std::setw(20) << name
                  << std::setw(15) << std::fixed << std::setprecision(2) << cpu_avg
                  << std::setw(15) << cuda_avg
                  << std::setw(14) << speedup << "x"
                  << std::setw(18) << std::setprecision(1) << throughput << std::endl;

        TestResult r;
        r.test_name = "Channel Count";
        r.configuration = name;
        r.time_ms = cuda_avg;
        r.speedup = speedup;
        r.throughput = throughput;
        results.push_back(r);
    }

    return results;
}

// ============================================================================
// Test 3: Resolution Comparison (Non-square, Video formats)
// ============================================================================

std::vector<TestResult> test_resolutions(const AdvancedTestConfig& config) {
    std::vector<TestResult> results;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 3: Resolution Comparison" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    ConvKernel kernel = get_kernel("gaussian", 5);

    std::vector<std::tuple<std::string, int, int>> resolutions = {
        {"VGA (640x480)", 640, 480},
        {"HD (1280x720)", 1280, 720},
        {"Full HD (1920x1080)", 1920, 1080},
        {"2K (2560x1440)", 2560, 1440},
        {"4K (3840x2160)", 3840, 2160},
        {"Square 1024x1024", 1024, 1024},
        {"Non-pow2 (1000x1000)", 1000, 1000},
        {"Non-pow2 (1234x567)", 1234, 567}
    };

    std::cout << "\nKernel: " << kernel.name << "\n\n";
    std::cout << std::setw(25) << "Resolution"
              << std::setw(12) << "Pixels (M)"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "CUDA (ms)"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(79, '-') << std::endl;

    for (const auto& [name, w, h] : resolutions) {
        Image image = get_test_image(w, h, 3);
        double megapixels = (w * h) / 1e6;

        // CPU timing
        std::vector<double> cpu_times;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cpu_timed(image, kernel, t);
            cpu_times.push_back(t);
        }
        double cpu_avg = compute_mean(cpu_times);

        // CUDA timing
        std::vector<double> cuda_times;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cuda_timed(image, kernel, t, CudaOptLevel::SHARED);
            cuda_times.push_back(t);
        }
        double cuda_avg = compute_mean(cuda_times);

        double speedup = cpu_avg / cuda_avg;

        std::cout << std::setw(25) << name
                  << std::setw(12) << std::fixed << std::setprecision(2) << megapixels
                  << std::setw(15) << std::setprecision(2) << cpu_avg
                  << std::setw(15) << cuda_avg
                  << std::setw(11) << speedup << "x" << std::endl;

        TestResult r;
        r.test_name = "Resolutions";
        r.configuration = name;
        r.time_ms = cuda_avg;
        r.speedup = speedup;
        r.throughput = megapixels / (cuda_avg / 1000.0);
        r.notes = std::to_string(w) + "x" + std::to_string(h);
        results.push_back(r);
    }

    return results;
}

// ============================================================================
// Test 4: Block Size Sweep (Occupancy Analysis)
// ============================================================================

std::vector<TestResult> test_block_sizes(const AdvancedTestConfig& config) {
    std::vector<TestResult> results;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 4: Block Size Occupancy Analysis" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int width = 2048, height = 2048;
    Image image = get_test_image(width, height, 3);
    ConvKernel kernel = get_kernel("gaussian", 5);

    std::vector<int> block_sizes = {4, 8, 12, 16, 20, 24, 28, 32};

    std::cout << "\nImage size: " << width << "x" << height << ", Kernel: " << kernel.name << "\n\n";
    std::cout << std::setw(15) << "Block Size"
              << std::setw(18) << "Threads/Block"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Throughput (MP/s)"
              << std::setw(15) << "Relative" << std::endl;
    std::cout << std::string(81, '-') << std::endl;

    double best_time = 1e9;
    int best_block = 0;

    for (int bs : block_sizes) {
        // CUDA timing
        std::vector<double> times;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cuda_timed(image, kernel, t, CudaOptLevel::SHARED, bs);
            times.push_back(t);
        }
        double avg_time = compute_mean(times);

        if (avg_time < best_time) {
            best_time = avg_time;
            best_block = bs;
        }

        int threads_per_block = bs * bs;
        double throughput = (width * height / 1e6) / (avg_time / 1000.0);
        double relative = best_time / avg_time;

        std::cout << std::setw(12) << bs << "x" << bs
                  << std::setw(15) << threads_per_block
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                  << std::setw(18) << std::setprecision(1) << throughput
                  << std::setw(14) << std::setprecision(2) << relative << "x" << std::endl;

        TestResult r;
        r.test_name = "Block Size";
        r.configuration = std::to_string(bs) + "x" + std::to_string(bs);
        r.time_ms = avg_time;
        r.speedup = relative;
        r.throughput = throughput;
        results.push_back(r);
    }

    std::cout << "\n>>> Optimal block size: " << best_block << "x" << best_block
              << " (" << best_block * best_block << " threads)" << std::endl;

    return results;
}

// ============================================================================
// Test 5: 2D vs Separable Convolution
// ============================================================================

std::vector<TestResult> test_separable_convolution(const AdvancedTestConfig& config) {
    std::vector<TestResult> results;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 5: 2D vs Separable Convolution" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int width = 1024, height = 1024;
    Image image = get_test_image(width, height, 3);

    std::vector<int> kernel_sizes = {3, 5, 7, 9, 11};

    std::cout << "\nImage size: " << width << "x" << height << "\n";
    std::cout << "Comparing 2D convolution O(K²) vs Separable O(2K)\n\n";
    std::cout << std::setw(10) << "Kernel"
              << std::setw(15) << "2D CPU (ms)"
              << std::setw(15) << "Sep CPU (ms)"
              << std::setw(15) << "Improvement"
              << std::setw(18) << "Theory (K²/2K)" << std::endl;
    std::cout << std::string(73, '-') << std::endl;

    for (int ks : kernel_sizes) {
        ConvKernel kernel = get_kernel("gaussian", ks);

        // Check if separable
        std::vector<float> h_kernel, v_kernel;
        bool separable = is_separable(kernel, h_kernel, v_kernel);

        if (!separable) {
            std::cout << std::setw(10) << (std::to_string(ks) + "x" + std::to_string(ks))
                      << " - Not separable -" << std::endl;
            continue;
        }

        // 2D convolution timing
        std::vector<double> times_2d;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cpu_timed(image, kernel, t);
            times_2d.push_back(t);
        }
        double avg_2d = compute_mean(times_2d);

        // Separable convolution timing
        std::vector<double> times_sep;
        for (int i = 0; i < config.iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            convolve_cpu_separable(image, h_kernel, v_kernel);
            auto end = std::chrono::high_resolution_clock::now();
            times_sep.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        double avg_sep = compute_mean(times_sep);

        double improvement = avg_2d / avg_sep;
        double theoretical = (ks * ks) / (2.0 * ks);

        std::cout << std::setw(10) << (std::to_string(ks) + "x" + std::to_string(ks))
                  << std::setw(15) << std::fixed << std::setprecision(2) << avg_2d
                  << std::setw(15) << avg_sep
                  << std::setw(14) << std::setprecision(2) << improvement << "x"
                  << std::setw(17) << std::setprecision(2) << theoretical << "x" << std::endl;

        TestResult r;
        r.test_name = "Separable Conv";
        r.configuration = std::to_string(ks) + "x" + std::to_string(ks);
        r.time_ms = avg_sep;
        r.speedup = improvement;
        r.throughput = (width * height / 1e6) / (avg_sep / 1000.0);
        r.notes = "Theoretical: " + std::to_string(theoretical) + "x";
        results.push_back(r);
    }

    return results;
}

// ============================================================================
// Test 6: Memory Bandwidth Analysis
// ============================================================================

std::vector<TestResult> test_memory_bandwidth(const AdvancedTestConfig& config) {
    std::vector<TestResult> results;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 6: Memory Bandwidth Analysis" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // GTX 1080 theoretical peak: 320 GB/s
    const double theoretical_bandwidth = 320.0;  // GB/s

    std::vector<std::tuple<int, int>> sizes = {
        {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}
    };

    ConvKernel kernel = get_kernel("gaussian", 3);  // Small kernel to be memory-bound

    std::cout << "\nTheoretical peak bandwidth (GTX 1080): " << theoretical_bandwidth << " GB/s\n";
    std::cout << "Kernel: " << kernel.name << " (small kernel = memory-bound)\n\n";
    std::cout << std::setw(15) << "Image Size"
              << std::setw(15) << "Data (MB)"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Bandwidth (GB/s)"
              << std::setw(15) << "Efficiency" << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    for (const auto& [w, h] : sizes) {
        Image image = get_test_image(w, h, 3);

        // Calculate data transferred:
        // Read: input image + kernel (kernel negligible)
        // Write: output image
        // Total: 2 * image_size (read + write)
        double data_mb = 2.0 * w * h * 3 / (1024.0 * 1024.0);  // MB
        double data_gb = data_mb / 1024.0;

        // CUDA timing
        std::vector<double> times;
        for (int i = 0; i < config.iterations; i++) {
            double t;
            convolve_cuda_timed(image, kernel, t, CudaOptLevel::SHARED);
            times.push_back(t);
        }
        double avg_time = compute_mean(times);

        // Calculate effective bandwidth
        double bandwidth = data_gb / (avg_time / 1000.0);  // GB/s
        double efficiency = 100.0 * bandwidth / theoretical_bandwidth;

        std::cout << std::setw(12) << w << "x" << h
                  << std::setw(15) << std::fixed << std::setprecision(2) << data_mb
                  << std::setw(15) << std::setprecision(3) << avg_time
                  << std::setw(18) << std::setprecision(1) << bandwidth
                  << std::setw(14) << std::setprecision(1) << efficiency << "%" << std::endl;

        TestResult r;
        r.test_name = "Bandwidth";
        r.configuration = std::to_string(w) + "x" + std::to_string(h);
        r.time_ms = avg_time;
        r.speedup = efficiency;  // Using speedup field for efficiency
        r.throughput = bandwidth;
        results.push_back(r);
    }

    std::cout << "\nNote: Actual bandwidth is lower due to kernel computation overhead\n";
    std::cout << "and non-coalesced memory access at image borders.\n";

    return results;
}

// ============================================================================
// Test 7: CUDA Streams (Multi-image processing)
// ============================================================================

std::vector<TestResult> test_cuda_streams(const AdvancedTestConfig& config) {
    std::vector<TestResult> results;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 7: Multi-Image Processing (Batch)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int width = 512, height = 512;
    ConvKernel kernel = get_kernel("gaussian", 5);

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};

    std::cout << "\nImage size: " << width << "x" << height << ", Kernel: " << kernel.name << "\n";
    std::cout << "Processing multiple images sequentially vs potential parallel\n\n";
    std::cout << std::setw(15) << "Batch Size"
              << std::setw(18) << "Total Time (ms)"
              << std::setw(18) << "Time/Image (ms)"
              << std::setw(18) << "Throughput (img/s)" << std::endl;
    std::cout << std::string(69, '-') << std::endl;

    double single_image_time = 0;

    for (int batch : batch_sizes) {
        // Create batch of images
        std::vector<Image> images;
        for (int i = 0; i < batch; i++) {
            images.push_back(get_test_image(width, height, 3));
        }

        // Time processing all images
        std::vector<double> times;
        for (int iter = 0; iter < config.iterations; iter++) {
            auto start = std::chrono::high_resolution_clock::now();

            for (const auto& img : images) {
                double t;
                convolve_cuda_timed(img, kernel, t, CudaOptLevel::SHARED);
            }

            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        double total_time = compute_mean(times);
        double time_per_image = total_time / batch;
        double throughput = batch / (total_time / 1000.0);  // images/second

        if (batch == 1) {
            single_image_time = time_per_image;
        }

        std::cout << std::setw(15) << batch
                  << std::setw(18) << std::fixed << std::setprecision(2) << total_time
                  << std::setw(18) << std::setprecision(3) << time_per_image
                  << std::setw(18) << std::setprecision(1) << throughput << std::endl;

        TestResult r;
        r.test_name = "Batch Processing";
        r.configuration = "Batch " + std::to_string(batch);
        r.time_ms = total_time;
        r.speedup = single_image_time / time_per_image;
        r.throughput = throughput;
        results.push_back(r);
    }

    std::cout << "\nNote: True CUDA streams would overlap computation and memory transfers.\n";
    std::cout << "This test shows sequential processing overhead.\n";

    return results;
}

// ============================================================================
// Run All Tests
// ============================================================================

std::vector<TestResult> run_all_advanced_tests(const AdvancedTestConfig& config) {
    std::vector<TestResult> all_results;

    std::cout << "\n" << std::string(70, '#') << std::endl;
    std::cout << "           ADVANCED PERFORMANCE TESTS" << std::endl;
    std::cout << std::string(70, '#') << std::endl;

    // Initialize base image (from directory or synthetic)
    init_base_image(config.images_dir);
    std::cout << std::endl;
    
    if (config.run_image_type_test) {
        auto r = test_image_types(config);
        all_results.insert(all_results.end(), r.begin(), r.end());
    }
    
    if (config.run_channels_test) {
        auto r = test_channel_counts(config);
        all_results.insert(all_results.end(), r.begin(), r.end());
    }
    
    if (config.run_resolution_test) {
        auto r = test_resolutions(config);
        all_results.insert(all_results.end(), r.begin(), r.end());
    }
    
    if (config.run_occupancy_test) {
        auto r = test_block_sizes(config);
        all_results.insert(all_results.end(), r.begin(), r.end());
    }
    
    if (config.run_separable_test) {
        auto r = test_separable_convolution(config);
        all_results.insert(all_results.end(), r.begin(), r.end());
    }
    
    if (config.run_bandwidth_test) {
        auto r = test_memory_bandwidth(config);
        all_results.insert(all_results.end(), r.begin(), r.end());
    }
    
    if (config.run_multistream_test) {
        auto r = test_cuda_streams(config);
        all_results.insert(all_results.end(), r.begin(), r.end());
    }
    
    std::cout << "\n" << std::string(70, '#') << std::endl;
    std::cout << "           ALL TESTS COMPLETE" << std::endl;
    std::cout << std::string(70, '#') << std::endl;
    
    return all_results;
}

// ============================================================================
// Export Functions
// ============================================================================

void export_advanced_results_csv(const std::vector<TestResult>& results,
                                  const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filepath << std::endl;
        return;
    }
    
    file << "test_name,configuration,time_ms,speedup,throughput,notes\n";
    
    for (const auto& r : results) {
        file << "\"" << r.test_name << "\","
             << "\"" << r.configuration << "\","
             << std::fixed << std::setprecision(4) << r.time_ms << ","
             << r.speedup << ","
             << r.throughput << ","
             << "\"" << r.notes << "\"\n";
    }
    
    file.close();
    std::cout << "\nResults exported to: " << filepath << std::endl;
}

void print_advanced_results(const std::vector<TestResult>& results) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "SUMMARY OF ALL ADVANCED TESTS" << std::endl;
    std::cout << std::string(60, '=') << "\n\n";
    
    std::string current_test;
    for (const auto& r : results) {
        if (r.test_name != current_test) {
            if (!current_test.empty()) std::cout << std::endl;
            std::cout << ">>> " << r.test_name << std::endl;
            current_test = r.test_name;
        }
        std::cout << "    " << std::setw(25) << std::left << r.configuration
                  << " | Time: " << std::setw(10) << std::right << std::fixed 
                  << std::setprecision(2) << r.time_ms << " ms"
                  << " | Speedup: " << std::setprecision(2) << r.speedup << "x"
                  << std::endl;
    }
}