#include "benchmark.h"
#include "cpu_convolution.h"
#include "gpu_convolution.cuh"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <set>





BenchmarkResult benchmark_single(const Image& image, const ConvKernel& kernel,
                                 const std::string& implementation,
                                 int iterations, int block_size) {

    BenchmarkResult result;
    result.image_width = image.width;
    result.image_height = image.height;
    result.kernel_size = kernel.size;
    result.kernel_name = kernel.name;
    result.implementation = implementation;
    result.block_size = block_size;

    std::vector<double> times;
    times.reserve(iterations);

    
    for (int i = 0; i < 2; i++) {
        if (implementation == "cpu") {
            double warmup;
            convolve_cpu_timed(image, kernel, warmup);
        } else {
            double warmup;
            CudaOptLevel opt = CudaOptLevel::SHARED;
            if (implementation == "cuda_global") opt = CudaOptLevel::GLOBAL;
            else if (implementation == "cuda_constant") opt = CudaOptLevel::CONSTANT;
            convolve_cuda_timed(image, kernel, warmup, opt, block_size);
        }
    }

    
    for (int i = 0; i < iterations; i++) {
        double time_ms;

        if (implementation == "cpu") {
            convolve_cpu_timed(image, kernel, time_ms);
        } else {
            CudaOptLevel opt = CudaOptLevel::SHARED;
            if (implementation == "cuda_global") opt = CudaOptLevel::GLOBAL;
            else if (implementation == "cuda_constant") opt = CudaOptLevel::CONSTANT;
            else if (implementation == "cuda_shared") opt = CudaOptLevel::SHARED;

            convolve_cuda_timed(image, kernel, time_ms, opt, block_size);
        }

        times.push_back(time_ms);
    }

    
    result.time_ms = compute_mean(times);
    result.stddev_ms = compute_stddev(times);
    result.min_time_ms = *std::min_element(times.begin(), times.end());
    result.max_time_ms = *std::max_element(times.begin(), times.end());

    
    double total_pixels = static_cast<double>(image.width * image.height);
    result.throughput_mpps = (total_pixels / 1e6) / (result.time_ms / 1000.0);

    return result;
}






static std::vector<std::string> find_kodak_images() {
    std::vector<std::string> paths;
    std::vector<std::string> search_dirs = {
        "images/input/kodak",
        "../images/input/kodak",
        "../../images/input/kodak"
    };

    for (const auto& dir : search_dirs) {
        for (int i = 1; i <= 24; i++) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/kodim%02d.png", dir.c_str(), i);
            std::ifstream f(filename);
            if (f.good()) {
                paths.push_back(filename);
            }
        }
        if (!paths.empty()) break;
    }

    return paths;
}


static Image resize_image(const Image& src, int target_size) {
    if (src.width == target_size && src.height == target_size) {
        return src;
    }

    Image resized(target_size, target_size, src.channels);

    float x_ratio = static_cast<float>(src.width) / target_size;
    float y_ratio = static_cast<float>(src.height) / target_size;

    for (int y = 0; y < target_size; y++) {
        for (int x = 0; x < target_size; x++) {
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

std::vector<BenchmarkResult> run_all_benchmarks(const BenchmarkConfig& config) {
    std::vector<BenchmarkResult> all_results;

    int total_tests = config.image_sizes.size() * config.kernel_sizes.size() *
                      config.kernel_types.size();
    int current_test = 0;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Benchmark Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    
    std::vector<std::string> kodak_paths = find_kodak_images();
    Image base_image;
    bool using_kodak = false;

    if (!kodak_paths.empty()) {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(0, kodak_paths.size() - 1);
        int idx = dis(gen);

        std::cout << "Found " << kodak_paths.size() << " Kodak images" << std::endl;
        std::cout << "Using: " << kodak_paths[idx] << " (index " << idx << ")" << std::endl;
        try {
            base_image = load_image(kodak_paths[idx]);
            using_kodak = true;
        } catch (...) {
            std::cout << "Failed to load, falling back to synthetic" << std::endl;
        }
    }

    if (!using_kodak) {
        std::cout << "Using synthetic images" << std::endl;
    }
    std::cout << std::endl;

    
    std::string base_name = "kodak";
    if (config.save_examples && using_kodak) {
        Image example_image = resize_image(base_image, 512);
        std::string orig_path = config.examples_dir + base_name + "_original.png";
        save_image(orig_path, example_image);
        std::cout << "Saved: " << orig_path << std::endl;
    }

    
    std::set<std::string> saved_examples;

    
    for (int img_size : config.image_sizes) {
        
        Image image;
        if (using_kodak) {
            image = resize_image(base_image, img_size);
        } else {
            image = create_test_image(img_size, img_size, 3, "gradient");
        }

        if (config.verbose) {
            std::cout << "Image size: " << img_size << "x" << img_size << std::endl;
        }

        
        for (const std::string& kernel_type : config.kernel_types) {
            
            for (int kernel_size : config.kernel_sizes) {
                current_test++;

                
                if (kernel_type == "sharpen" && kernel_size != 3) continue;
                if (kernel_type == "sobel_x" && kernel_size != 3) continue;
                if (kernel_type == "sobel_y" && kernel_size != 3) continue;

                ConvKernel kernel = get_kernel(kernel_type, kernel_size);

                if (config.verbose) {
                    std::cout << "  Testing: " << kernel.name
                              << " [" << current_test << "/" << total_tests << "]" << std::endl;
                }

                if (config.save_examples && img_size == 512) {
                    std::string example_key = kernel_type + "_" + std::to_string(kernel_size);
                    if (saved_examples.find(example_key) == saved_examples.end()) {
                        Image output = convolve_cuda(image, kernel, CudaOptLevel::SHARED);
                        std::string out_path = config.examples_dir + base_name + "_" +
                                               kernel_type + "_" + std::to_string(kernel_size) + "x" +
                                               std::to_string(kernel_size) + ".png";
                        save_image(out_path, output);
                        std::cout << "  Saved example: " << out_path << std::endl;
                        saved_examples.insert(example_key);
                    }
                }

                
                BenchmarkResult cpu_result;
                if (config.test_cpu) {
                    cpu_result = benchmark_single(image, kernel, "cpu",
                                                  config.iterations, 16);
                    all_results.push_back(cpu_result);
                }

                
                for (int block_size : config.block_sizes) {
                    if (config.test_cuda_global) {
                        BenchmarkResult result = benchmark_single(image, kernel,
                                                                  "cuda_global",
                                                                  config.iterations,
                                                                  block_size);
                        if (config.test_cpu) {
                            result.speedup = cpu_result.time_ms / result.time_ms;
                        }
                        all_results.push_back(result);
                    }

                    if (config.test_cuda_constant) {
                        BenchmarkResult result = benchmark_single(image, kernel,
                                                                  "cuda_constant",
                                                                  config.iterations,
                                                                  block_size);
                        if (config.test_cpu) {
                            result.speedup = cpu_result.time_ms / result.time_ms;
                        }
                        all_results.push_back(result);
                    }

                    if (config.test_cuda_shared) {
                        BenchmarkResult result = benchmark_single(image, kernel,
                                                                  "cuda_shared",
                                                                  config.iterations,
                                                                  block_size);
                        if (config.test_cpu) {
                            result.speedup = cpu_result.time_ms / result.time_ms;
                        }
                        all_results.push_back(result);
                    }
                }
            }
        }
    }

    
    std::string csv_path = config.output_dir + "benchmark_results.csv";
    export_results_csv(all_results, csv_path);

    std::cout << "\nBenchmark complete. Results saved to: " << csv_path << std::endl;

    return all_results;
}

std::vector<BenchmarkResult> run_all_benchmarks(const std::string& output_dir) {
    BenchmarkConfig config;
    config.output_dir = output_dir;
    return run_all_benchmarks(config);
}





std::map<std::string, double> compare_implementations(
    const Image& image, const ConvKernel& kernel, int iterations) {
    
    std::map<std::string, double> speedups;
    
    
    BenchmarkResult cpu = benchmark_single(image, kernel, "cpu", iterations);
    speedups["cpu"] = 1.0;
    
    
    std::vector<std::string> cuda_impls = {"cuda_global", "cuda_constant", "cuda_shared"};
    
    for (const auto& impl : cuda_impls) {
        BenchmarkResult result = benchmark_single(image, kernel, impl, iterations);
        speedups[impl] = cpu.time_ms / result.time_ms;
    }
    
    return speedups;
}





void export_results_csv(const std::vector<BenchmarkResult>& results,
                        const std::string& filepath) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    
    
    file << "image_size,image_width,image_height,kernel_size,kernel_name,"
         << "implementation,block_size,time_ms,stddev_ms,min_time_ms,max_time_ms,"
         << "speedup,throughput_mpps\n";
    
    
    for (const auto& r : results) {
        file << r.image_width << ","
             << r.image_width << ","
             << r.image_height << ","
             << r.kernel_size << ","
             << "\"" << r.kernel_name << "\","
             << r.implementation << ","
             << r.block_size << ","
             << std::fixed << std::setprecision(4)
             << r.time_ms << ","
             << r.stddev_ms << ","
             << r.min_time_ms << ","
             << r.max_time_ms << ","
             << r.speedup << ","
             << r.throughput_mpps << "\n";
    }
    
    file.close();
}

void print_result(const BenchmarkResult& result) {
    std::cout << std::setw(10) << result.image_width << "x" << result.image_height
              << " | " << std::setw(15) << result.kernel_name
              << " | " << std::setw(14) << result.implementation
              << " | " << std::setw(6) << result.block_size
              << " | " << std::setw(10) << std::fixed << std::setprecision(3) 
              << result.time_ms << " ms"
              << " | " << std::setw(8) << std::setprecision(2) 
              << result.speedup << "x"
              << std::endl;
}

void print_results_summary(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Results Summary" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << std::setw(12) << "Image Size"
              << " | " << std::setw(15) << "Kernel"
              << " | " << std::setw(14) << "Implementation"
              << " | " << std::setw(6) << "Block"
              << " | " << std::setw(14) << "Time"
              << " | " << std::setw(10) << "Speedup"
              << std::endl;
    
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& r : results) {
        print_result(r);
    }
    
    
    auto best = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.speedup < b.speedup;
        });
    
    if (best != results.end() && best->speedup > 1.0) {
        std::cout << "\nBest speedup: " << std::fixed << std::setprecision(2) 
                  << best->speedup << "x"
                  << " (" << best->implementation 
                  << ", " << best->image_width << "x" << best->image_height
                  << ", " << best->kernel_name << ")" << std::endl;
    }
}





bool validate_cuda_correctness(const Image& image, const ConvKernel& kernel,
                               int tolerance) {
    
    
    Image cpu_output = convolve_cpu(image, kernel);
    
    
    Image cuda_output = convolve_cuda(image, kernel, CudaOptLevel::SHARED);
    
    
    return images_equal(cpu_output, cuda_output, tolerance);
}

std::map<std::string, bool> validate_all_implementations(
    const Image& image, const ConvKernel& kernel) {
    
    std::map<std::string, bool> validation;
    
    
    Image cpu_output = convolve_cpu(image, kernel);
    validation["cpu"] = true;
    
    
    std::vector<std::pair<std::string, CudaOptLevel>> impls = {
        {"cuda_global", CudaOptLevel::GLOBAL},
        {"cuda_constant", CudaOptLevel::CONSTANT},
        {"cuda_shared", CudaOptLevel::SHARED}
    };
    
    for (const auto& [name, opt] : impls) {
        Image cuda_output = convolve_cuda(image, kernel, opt);
        validation[name] = images_equal(cpu_output, cuda_output, 1);
    }
    
    return validation;
}
