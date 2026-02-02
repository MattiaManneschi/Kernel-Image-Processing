#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>
#include <map>
#include "image_io.h"
#include "kernels.h"





struct BenchmarkResult {
    
    int image_width;
    int image_height;
    int kernel_size;
    std::string kernel_name;
    std::string implementation;  
    int block_size;
    
    
    double time_ms;          
    double stddev_ms;        
    double min_time_ms;      
    double max_time_ms;      
    
    
    double speedup;          
    double throughput_mpps;  
    
    BenchmarkResult() : image_width(0), image_height(0), kernel_size(0),
                        block_size(16), time_ms(0), stddev_ms(0),
                        min_time_ms(0), max_time_ms(0), speedup(1.0),
                        throughput_mpps(0) {}
};





struct BenchmarkConfig {
    
    std::vector<int> image_sizes = {256, 512, 1024, 2048, 4096};
    
    
    std::vector<int> kernel_sizes = {3, 5, 7};
    
    
    std::vector<std::string> kernel_types = {"gaussian", "box", "sharpen", "sobel_x"};
    
    
    std::vector<int> block_sizes = {8, 16, 32};
    
    
    int iterations = 10;
    
    
    int warmup_runs = 2;
    
    
    std::string output_dir = "results/benchmarks/";
    
    
    bool test_cpu = true;
    bool test_cuda_global = true;
    bool test_cuda_constant = true;
    bool test_cuda_shared = true;
    
    
    bool use_synthetic_images = true;
    
    
    bool save_examples = false;
    std::string examples_dir = "images/output/";

    
    bool verbose = true;
};





/**
 * Run comprehensive benchmarks with all configurations
 * 
 * @param config Benchmark configuration
 * @return Vector of all benchmark results
 */
std::vector<BenchmarkResult> run_all_benchmarks(const BenchmarkConfig& config);

/**
 * Run benchmarks with default configuration
 * 
 * @param output_dir Output directory for results
 * @return Vector of benchmark results
 */
std::vector<BenchmarkResult> run_all_benchmarks(const std::string& output_dir = "results/benchmarks/");

/**
 * Benchmark single configuration
 * 
 * @param image Input image
 * @param kernel Convolution kernel
 * @param implementation Which implementation to test
 * @param iterations Number of iterations
 * @param block_size CUDA block size (for CUDA implementations)
 * @return Benchmark result
 */
BenchmarkResult benchmark_single(const Image& image, const ConvKernel& kernel,
                                 const std::string& implementation,
                                 int iterations = 10, int block_size = 16);

/**
 * Run speedup comparison (CPU vs all CUDA implementations)
 * 
 * @param image Input image
 * @param kernel Convolution kernel
 * @param iterations Number of iterations
 * @return Map of implementation name to speedup factor
 */
std::map<std::string, double> compare_implementations(
    const Image& image, const ConvKernel& kernel, int iterations = 10);





/**
 * Export results to CSV file
 * 
 * @param results Benchmark results
 * @param filepath Output CSV path
 */
void export_results_csv(const std::vector<BenchmarkResult>& results,
                        const std::string& filepath);

/**
 * Print results summary to console
 * 
 * @param results Benchmark results
 */
void print_results_summary(const std::vector<BenchmarkResult>& results);

/**
 * Print single result to console
 * 
 * @param result Benchmark result
 */
void print_result(const BenchmarkResult& result);





/**
 * Validate CUDA implementation against CPU (correctness check)
 * 
 * @param image Input image
 * @param kernel Convolution kernel
 * @param tolerance Maximum allowed pixel difference
 * @return true if CUDA output matches CPU within tolerance
 */
bool validate_cuda_correctness(const Image& image, const ConvKernel& kernel,
                               int tolerance = 1);

/**
 * Run validation for all implementations
 * 
 * @param image Input image
 * @param kernel Convolution kernel
 * @return Map of implementation name to validation status
 */
std::map<std::string, bool> validate_all_implementations(
    const Image& image, const ConvKernel& kernel);

#endif 
