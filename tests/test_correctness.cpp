/**
 * Test suite for validating CUDA implementations against CPU reference
 */

#include <iostream>
#include <cstdlib>
#include <cmath>

#include "../src/image_io.h"
#include "../src/kernels.h"
#include "../src/cpu_convolution.h"
#include "../src/gpu_convolution.cuh"
#include "../src/utils.h"

// Test configuration
const int TEST_SIZES[] = {64, 128, 256, 512};
const int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);
const int TOLERANCE = 2;  // Max allowed pixel difference

// Test result
struct TestResult {
    std::string test_name;
    bool passed;
    double max_diff;
    double psnr;
};

void print_result(const TestResult& result) {
    std::cout << (result.passed ? "[PASS]" : "[FAIL]") << " "
              << result.test_name;
    if (!result.passed) {
        std::cout << " (max_diff=" << result.max_diff 
                  << ", PSNR=" << result.psnr << " dB)";
    }
    std::cout << std::endl;
}

TestResult test_implementation(const Image& image, const ConvKernel& kernel,
                               const std::string& impl_name, CudaOptLevel opt_level) {
    TestResult result;
    result.test_name = impl_name + " - " + kernel.name + " (" + 
                       std::to_string(image.width) + "x" + std::to_string(image.height) + ")";
    
    // CPU reference
    Image cpu_output = convolve_cpu(image, kernel);
    
    // CUDA implementation
    Image cuda_output = convolve_cuda(image, kernel, opt_level);
    
    // Compare
    result.max_diff = 0;
    for (size_t i = 0; i < cpu_output.data.size(); i++) {
        double diff = std::abs(static_cast<int>(cpu_output.data[i]) - 
                              static_cast<int>(cuda_output.data[i]));
        result.max_diff = std::max(result.max_diff, diff);
    }
    
    result.psnr = compute_psnr(cpu_output, cuda_output);
    result.passed = (result.max_diff <= TOLERANCE);
    
    return result;
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Kernel Image Processing - Test Suite\n";
    std::cout << "========================================\n\n";
    
    // Check CUDA availability
    if (!cuda_available()) {
        std::cerr << "ERROR: No CUDA device available\n";
        return 1;
    }
    
    std::vector<TestResult> all_results;
    int passed = 0, failed = 0;
    
    // Test kernels
    std::vector<ConvKernel> kernels = {
        Kernels::gaussian_3x3(),
        Kernels::box_blur_3x3(),
        Kernels::sharpen_3x3(),
        Kernels::sobel_x_3x3(),
        Kernels::gaussian_5x5()
    };
    
    // Test CUDA implementations
    std::vector<std::pair<std::string, CudaOptLevel>> implementations = {
        {"CUDA Global", CudaOptLevel::GLOBAL},
        {"CUDA Constant", CudaOptLevel::CONSTANT},
        {"CUDA Shared", CudaOptLevel::SHARED}
    };
    
    // Run tests
    for (int size : TEST_SIZES) {
        std::cout << "Testing " << size << "x" << size << " images...\n";
        
        // Create test image
        Image image = create_test_image(size, size, 3, "gradient");
        
        for (const auto& kernel : kernels) {
            for (const auto& [impl_name, opt_level] : implementations) {
                TestResult result = test_implementation(image, kernel, impl_name, opt_level);
                all_results.push_back(result);
                
                if (result.passed) passed++;
                else failed++;
                
                print_result(result);
            }
        }
        std::cout << "\n";
    }
    
    // Summary
    std::cout << "========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    std::cout << "Total:  " << (passed + failed) << "\n";
    std::cout << "\n";
    
    if (failed > 0) {
        std::cout << "SOME TESTS FAILED!\n";
        return 1;
    }
    
    std::cout << "ALL TESTS PASSED!\n";
    return 0;
}
