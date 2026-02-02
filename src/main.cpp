#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "image_io.h"
#include "kernels.h"
#include "cpu_convolution.h"
#include "gpu_convolution.cuh"
#include "benchmark.h"
#include "advanced_tests.h"
#include "utils.h"

struct Options
{
    std::string input_path;
    std::string output_path;
    std::string kernel_type = "gaussian";
    int kernel_size = 3;
    std::string mode = "both";
    int block_size = 16;
    bool run_benchmark = false;
    std::string benchmark_output = "results/benchmarks/";
    bool verbose = false;
    bool show_info = false;
    bool validate = false;
    bool list_kernels = false;
    bool save_examples = false;
    bool run_advanced_tests = false;
    std::string images_dir = "";
    bool generate_all = false;
    std::string output_dir_all = "images/output/all_filters/";
};

void print_usage(const char* program)
{
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
    std::cout << "  --save-examples        Save example output images (with --benchmark)\n";
    std::cout << "  --advanced-tests       Run advanced performance tests\n";
    std::cout << "  --generate-all         Apply ALL kernels to input image and save results\n";
    std::cout << "  --images-dir <path>    Directory with test images (e.g., images/input/kodak)\n";
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

Options parse_args(int argc, char* argv[])
{
    Options opts;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-i" || arg == "--input")
        {
            if (i + 1 < argc) opts.input_path = argv[++i];
        }
        else if (arg == "-o" || arg == "--output")
        {
            if (i + 1 < argc) opts.output_path = argv[++i];
        }
        else if (arg == "-k" || arg == "--kernel")
        {
            if (i + 1 < argc) opts.kernel_type = argv[++i];
        }
        else if (arg == "-s" || arg == "--size")
        {
            if (i + 1 < argc) opts.kernel_size = std::stoi(argv[++i]);
        }
        else if (arg == "-m" || arg == "--mode")
        {
            if (i + 1 < argc) opts.mode = argv[++i];
        }
        else if (arg == "-b" || arg == "--blocksize")
        {
            if (i + 1 < argc) opts.block_size = std::stoi(argv[++i]);
        }
        else if (arg == "--benchmark")
        {
            opts.run_benchmark = true;
        }
        else if (arg == "--save-examples")
        {
            opts.save_examples = true;
        }
        else if (arg == "--advanced-tests")
        {
            opts.run_advanced_tests = true;
        }
        else if (arg == "--generate-all")
        {
            opts.generate_all = true;
        }
        else if (arg == "--images-dir")
        {
            if (i + 1 < argc) opts.images_dir = argv[++i];
        }
        else if (arg == "--output-dir")
        {
            if (i + 1 < argc) opts.benchmark_output = argv[++i];
        }
        else if (arg == "--validate")
        {
            opts.validate = true;
        }
        else if (arg == "--info")
        {
            opts.show_info = true;
        }
        else if (arg == "--list-kernels")
        {
            opts.list_kernels = true;
        }
        else if (arg == "-v" || arg == "--verbose")
        {
            opts.verbose = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    return opts;
}

int main(int argc, char* argv[])
{
    Options opts = parse_args(argc, argv);

    if (opts.verbose)
    {
        Logger::instance().set_verbose(true);
    }

    if (opts.show_info)
    {
        print_cuda_info();
        return 0;
    }

    if (opts.list_kernels)
    {
        std::cout << "Available kernels:\n";
        for (const auto& k : get_available_kernels())
        {
            std::cout << "  - " << k << "\n";
        }
        std::cout << "\nKernel sizes: 3, 5, 7 (not all kernels support all sizes)\n";
        return 0;
    }

    if (opts.run_benchmark)
    {
        std::cout << "Running comprehensive benchmarks...\n";

        BenchmarkConfig config;
        config.output_dir = opts.benchmark_output;
        config.verbose = opts.verbose;
        config.save_examples = opts.save_examples;

        auto results = run_all_benchmarks(config);
        print_results_summary(results);

        return 0;
    }

    if (opts.run_advanced_tests)
    {
        std::cout << "Running advanced performance tests...\n";

        AdvancedTestConfig config;
        config.verbose = opts.verbose;
        config.output_dir = "results/advanced_tests/";
        config.images_dir = opts.images_dir;

        auto results = run_all_advanced_tests(config);
        export_advanced_results_csv(results, config.output_dir + "advanced_results.csv");
        print_advanced_results(results);

        return 0;
    }

    if (opts.generate_all)
    {
        if (opts.input_path.empty())
        {
            std::cerr << "Error: Input image required for --generate-all (use -i)\n";
            return 1;
        }

        std::cout << "\n";
        std::cout << "============================================\n";
        std::cout << "  Generate All Filtered Images\n";
        std::cout << "============================================\n\n";


        std::cout << "Loading: " << opts.input_path << "\n";
        Image image = load_image(opts.input_path);
        std::cout << "Size: " << image.width << "x" << image.height << ", " << image.channels << " channels\n\n";


        std::string img_name = opts.input_path;

        size_t last_slash = img_name.find_last_of("/\\");
        if (last_slash != std::string::npos)
        {
            img_name = img_name.substr(last_slash + 1);
        }

        size_t last_dot = img_name.find_last_of(".");
        if (last_dot != std::string::npos)
        {
            img_name = img_name.substr(0, last_dot);
        }


        std::string out_dir = "images/output/" + img_name + "/";
        std::string mkdir_cmd = "mkdir -p " + out_dir;
        system(mkdir_cmd.c_str());


        std::string orig_path = out_dir + "00_original.png";
        save_image(orig_path, image);
        std::cout << "[OK] " << orig_path << " (Original)\n";


        struct KernelConfig
        {
            std::string name;
            std::string type;
            int size;
            std::string description;
        };

        std::vector<KernelConfig> kernels = {
            {"01_identity", "identity", 3, "Identity (no change)"},
            {"02_box_blur_3x3", "box", 3, "Box Blur 3x3"},
            {"03_box_blur_5x5", "box", 5, "Box Blur 5x5"},
            {"04_box_blur_7x7", "box", 7, "Box Blur 7x7"},
            {"05_gaussian_3x3", "gaussian", 3, "Gaussian Blur 3x3"},
            {"06_gaussian_5x5", "gaussian", 5, "Gaussian Blur 5x5"},
            {"07_gaussian_7x7", "gaussian", 7, "Gaussian Blur 7x7"},
            {"08_sharpen", "sharpen", 3, "Sharpen"},
            {"09_sharpen_strong", "sharpen_strong", 3, "Sharpen Strong"},
            {"10_edge_sobel_x", "sobel_x", 3, "Sobel X (horizontal edges)"},
            {"11_edge_sobel_y", "sobel_y", 3, "Sobel Y (vertical edges)"},
            {"12_edge_prewitt_x", "prewitt_x", 3, "Prewitt X"},
            {"13_edge_prewitt_y", "prewitt_y", 3, "Prewitt Y"},
            {"14_edge_laplacian", "laplacian", 3, "Laplacian (all edges)"},
            {"15_edge_laplacian_diag", "laplacian_diag", 3, "Laplacian with diagonals"},
            {"16_emboss", "emboss", 3, "Emboss"}
        };

        std::cout << "\nApplying " << kernels.size() << " filters...\n\n";

        for (const auto& kc : kernels)
        {
            try
            {
                ConvKernel kernel = get_kernel(kc.type, kc.size);

                double time_ms;
                Image output = convolve_cuda_timed(image, kernel, time_ms, CudaOptLevel::SHARED);

                std::string out_path = out_dir + kc.name + ".png";
                save_image(out_path, output);

                std::cout << "[OK] " << out_path << " (" << kc.description << ") - "
                    << std::fixed << std::setprecision(2) << time_ms << " ms\n";
            }
            catch (const std::exception& e)
            {
                std::cout << "[FAIL] " << kc.name << ": " << e.what() << "\n";
            }
        }

        std::cout << "\n============================================\n";
        std::cout << "  All images saved to: " << out_dir << "\n";
        std::cout << "============================================\n\n";

        return 0;
    }


    if (opts.validate)
    {
        std::cout << "Running validation tests...\n";

        Image test_image = create_test_image(256, 256, 3, "gradient");
        ConvKernel kernel = get_kernel(opts.kernel_type, opts.kernel_size);

        auto validation = validate_all_implementations(test_image, kernel);

        std::cout << "\nValidation results:\n";
        for (const auto& [impl, valid] : validation)
        {
            std::cout << "  " << impl << ": "
                << (valid ? "PASS" : "FAIL") << "\n";
        }

        return 0;
    }


    if (opts.input_path.empty())
    {
        std::cerr << "Error: Input image required (use -i or --input)\n";
        print_usage(argv[0]);
        return 1;
    }

    try
    {
        std::cout << "Loading image: " << opts.input_path << "\n";
        Image image = load_image(opts.input_path);
        print_image_info(image, "Input");


        ConvKernel kernel = get_kernel(opts.kernel_type, opts.kernel_size);
        std::cout << "Kernel: " << kernel.name << "\n";

        if (opts.verbose)
        {
            kernel.print();
        }

        Image output;
        double cpu_time = 0, cuda_time = 0;


        if (opts.mode == "cpu" || opts.mode == "both")
        {
            std::cout << "\n=== CPU Processing ===\n";

            output = convolve_cpu_timed(image, kernel, cpu_time);

            std::cout << "Time: " << format_time(cpu_time) << "\n";

            if (!opts.output_path.empty() && opts.mode == "cpu")
            {
                save_image(opts.output_path, output);
                std::cout << "Saved: " << opts.output_path << "\n";
            }
        }


        if (opts.mode == "cuda" || opts.mode == "both")
        {
            std::cout << "\n=== CUDA Processing ===\n";
            std::cout << "Block size: " << opts.block_size << "x" << opts.block_size << "\n";

            output = convolve_cuda_timed(image, kernel, cuda_time,
                                         CudaOptLevel::SHARED, opts.block_size);

            std::cout << "Time: " << format_time(cuda_time) << "\n";

            if (opts.mode == "both" && cpu_time > 0)
            {
                double speedup = cpu_time / cuda_time;
                std::cout << "Speedup: " << format_speedup(speedup) << "\n";
            }

            if (!opts.output_path.empty())
            {
                save_image(opts.output_path, output);
                std::cout << "Saved: " << opts.output_path << "\n";
            }
        }

        std::cout << "\nDone!\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
