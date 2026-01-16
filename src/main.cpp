#include <iostream>
#include <string>
#include <cstring>
#include <vector>

#include "image_io.h"
#include "kernels.h"
#include "cpu_convolution.h"
#include "gpu_convolution.cuh"
#include "benchmark.h"
#include "utils.h"

// ============================================================================
// Command Line Options
// ============================================================================

struct Options {
    std::string input_path;
    std::string output_path;
    std::string kernel_type = "gaussian";
    int kernel_size = 3;
    std::string mode = "both";  // cpu, cuda, both
    int block_size = 16;
    bool run_benchmark = false;
    std::string benchmark_output = "results/benchmarks/";
    bool verbose = false;
    bool show_info = false;
    bool validate = false;
    bool list_kernels = false;
    bool save_examples = false;
};

void print_usage(const char* program) {
    std::cout << "\n";
    std::cout << "Kernel Image Processing - CUDA\n";
    std::cout << "==============================\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input <path>     Input image path\n";
    std::cout << "  -o, --output <path>    Output image path\n";
    std::cout << "  -k, --kernel <type>    Kernel type (default: gaussian)\n";
    std::cout << "  -s, --size <n>         Kernel size: 3, 5, 7 (default: 3)\n";
    std::cout << "  -m, --mode <mode>      Execution mode: cpu, cuda, both (default: both)\n";
    std::cout << "  -b, --blocksize <n>    CUDA block size: 8, 16, 32 (default: 16)\n";
    std::cout << "  --benchmark            Run comprehensive benchmarks\n";
    std::cout << "  --output-dir <path>    Benchmark output directory\n";
    std::cout << "  --validate             Validate CUDA output against CPU\n";
    std::cout << "  --info                 Show CUDA device info\n";
    std::cout << "  --list-kernels         List available kernels\n";
    std::cout << "  -v, --verbose          Verbose output\n";
    std::cout << "  -h, --help             Show this help\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " -i input.png -o output.png -k gaussian -s 5\n";
    std::cout << "  " << program << " -i input.png -o output.png -k sobel_x -m cuda\n";
    std::cout << "  " << program << " --benchmark\n";
    std::cout << "  " << program << " --info\n";
    std::cout << "\n";
}

Options parse_args(int argc, char* argv[]) {
    Options opts;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) opts.input_path = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) opts.output_path = argv[++i];
        } else if (arg == "-k" || arg == "--kernel") {
            if (i + 1 < argc) opts.kernel_type = argv[++i];
        } else if (arg == "-s" || arg == "--size") {
            if (i + 1 < argc) opts.kernel_size = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--mode") {
            if (i + 1 < argc) opts.mode = argv[++i];
        } else if (arg == "-b" || arg == "--blocksize") {
            if (i + 1 < argc) opts.block_size = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            opts.run_benchmark = true;
        } else if (arg == "--save-examples") {
            opts.save_examples = true;
        } else if (arg == "--output-dir") {
            if (i + 1 < argc) opts.benchmark_output = argv[++i];
        } else if (arg == "--validate") {
            opts.validate = true;
        } else if (arg == "--info") {
            opts.show_info = true;
        } else if (arg == "--list-kernels") {
            opts.list_kernels = true;
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return opts;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Options opts = parse_args(argc, argv);
    
    // Set verbosity
    if (opts.verbose) {
        Logger::instance().set_verbose(true);
    }
    
    // Show CUDA info
    if (opts.show_info) {
        print_cuda_info();
        return 0;
    }
    
    // List available kernels
    if (opts.list_kernels) {
        std::cout << "Available kernels:\n";
        for (const auto& k : get_available_kernels()) {
            std::cout << "  - " << k << "\n";
        }
        std::cout << "\nKernel sizes: 3, 5, 7 (not all kernels support all sizes)\n";
        return 0;
    }
    
    // Run benchmarks
    if (opts.run_benchmark) {
        std::cout << "Running comprehensive benchmarks...\n";
        
        BenchmarkConfig config;
        config.output_dir = opts.benchmark_output;
        config.verbose = opts.verbose;
        
        auto results = run_all_benchmarks(config);
        print_results_summary(results);
        
        return 0;
    }
    
    // Validation mode
    if (opts.validate) {
        std::cout << "Running validation tests...\n";
        
        Image test_image = create_test_image(256, 256, 3, "gradient");
        ConvKernel kernel = get_kernel(opts.kernel_type, opts.kernel_size);
        
        auto validation = validate_all_implementations(test_image, kernel);
        
        std::cout << "\nValidation results:\n";
        for (const auto& [impl, valid] : validation) {
            std::cout << "  " << impl << ": " 
                      << (valid ? "PASS" : "FAIL") << "\n";
        }
        
        return 0;
    }
    
    // Single image processing
    if (opts.input_path.empty()) {
        std::cerr << "Error: Input image required (use -i or --input)\n";
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        // Load image
        std::cout << "Loading image: " << opts.input_path << "\n";
        Image image = load_image(opts.input_path);
        print_image_info(image, "Input");
        
        // Get kernel
        ConvKernel kernel = get_kernel(opts.kernel_type, opts.kernel_size);
        std::cout << "Kernel: " << kernel.name << "\n";
        
        if (opts.verbose) {
            kernel.print();
        }
        
        Image output;
        double cpu_time = 0, cuda_time = 0;
        
        // CPU processing
        if (opts.mode == "cpu" || opts.mode == "both") {
            std::cout << "\n=== CPU Processing ===\n";
            
            output = convolve_cpu_timed(image, kernel, cpu_time);
            
            std::cout << "Time: " << format_time(cpu_time) << "\n";
            
            if (!opts.output_path.empty() && opts.mode == "cpu") {
                save_image(opts.output_path, output);
                std::cout << "Saved: " << opts.output_path << "\n";
            }
        }
        
        // CUDA processing
        if (opts.mode == "cuda" || opts.mode == "both") {
            std::cout << "\n=== CUDA Processing ===\n";
            std::cout << "Block size: " << opts.block_size << "x" << opts.block_size << "\n";
            
            output = convolve_cuda_timed(image, kernel, cuda_time, 
                                         CudaOptLevel::SHARED, opts.block_size);
            
            std::cout << "Time: " << format_time(cuda_time) << "\n";
            
            if (opts.mode == "both" && cpu_time > 0) {
                double speedup = cpu_time / cuda_time;
                std::cout << "Speedup: " << format_speedup(speedup) << "\n";
            }
            
            if (!opts.output_path.empty()) {
                save_image(opts.output_path, output);
                std::cout << "Saved: " << opts.output_path << "\n";
            }
        }
        
        std::cout << "\nDone!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
