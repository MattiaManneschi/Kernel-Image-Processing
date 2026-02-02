#ifndef ADVANCED_TESTS_H
#define ADVANCED_TESTS_H

#include <string>
#include <vector>
#include <map>
#include "image_io.h"
#include "kernels.h"





struct TestResult {
    std::string test_name;
    std::string configuration;
    double time_ms;
    double speedup;
    double throughput;  
    std::string notes;
};

struct AdvancedTestConfig {
    bool run_image_type_test = true;      
    bool run_channels_test = true;         
    bool run_resolution_test = true;       
    bool run_occupancy_test = true;        
    bool run_separable_test = true;        
    bool run_bandwidth_test = true;        
    bool run_multistream_test = true;      

    int iterations = 10;
    std::string output_dir = "results/advanced_tests/";
    std::string images_dir = "";           
    bool verbose = true;
};





/**
 * Test 1: Compare performance on different image types
 * - Natural images (Kodak)
 * - Synthetic patterns (gradient, checker, noise, uniform)
 */
std::vector<TestResult> test_image_types(const AdvancedTestConfig& config);

/**
 * Test 2: Compare performance across different channel counts
 * - Grayscale (1 channel)
 * - RGB (3 channels)
 * - RGBA (4 channels)
 */
std::vector<TestResult> test_channel_counts(const AdvancedTestConfig& config);

/**
 * Test 3: Test non-square and standard video resolutions
 * - 1280x720 (HD)
 * - 1920x1080 (Full HD)
 * - 3840x2160 (4K)
 * - Non-power-of-2 dimensions
 */
std::vector<TestResult> test_resolutions(const AdvancedTestConfig& config);

/**
 * Test 4: Block size occupancy sweep
 * - Test block sizes: 4, 8, 12, 16, 20, 24, 28, 32
 * - Find optimal configuration
 */
std::vector<TestResult> test_block_sizes(const AdvancedTestConfig& config);

/**
 * Test 5: Compare 2D convolution vs separable convolution
 * - Only for separable kernels (Gaussian, Box)
 * - Show O(K²) vs O(2K) complexity
 */
std::vector<TestResult> test_separable_convolution(const AdvancedTestConfig& config);

/**
 * Test 6: Memory bandwidth analysis
 * - Measure effective bandwidth (GB/s)
 * - Compare to theoretical peak (GTX 1080: 320 GB/s)
 */
std::vector<TestResult> test_memory_bandwidth(const AdvancedTestConfig& config);

/**
 * Test 7: Multi-stream CUDA processing
 * - Process multiple images concurrently
 * - Overlap compute and memory transfers
 */
std::vector<TestResult> test_cuda_streams(const AdvancedTestConfig& config);





/**
 * Run all advanced tests
 */
std::vector<TestResult> run_all_advanced_tests(const AdvancedTestConfig& config);

/**
 * Export results to CSV
 */
void export_advanced_results_csv(const std::vector<TestResult>& results,
                                  const std::string& filepath);

/**
 * Print results summary
 */
void print_advanced_results(const std::vector<TestResult>& results);

#endif 