# Kernel Image Processing - CUDA

Parallel implementation of convolution-based image filters using CUDA.

## Project for Parallel Programming Course
University of Florence - Academic Year 2024/2025

## Author
- [Your Name] ([Your Student ID])

## Description

This project implements several convolution kernels for image processing, comparing sequential CPU implementation with parallel CUDA implementation on GPU.

### Implemented Filters
- **Box Blur** (3x3, 5x5, 7x7) - Simple averaging filter
- **Gaussian Blur** (3x3, 5x5) - Weighted blur for smoother results
- **Sharpen** - Edge enhancement filter
- **Sobel Edge Detection** (X and Y) - Gradient-based edge detection

### CUDA Optimizations
- Global Memory baseline
- Constant Memory for kernel coefficients
- Shared Memory for tile-based processing
- Loop unrolling

## Requirements

- CUDA Toolkit >= 11.0
- GCC >= 9.0
- NVIDIA GPU with Compute Capability >= 6.1 (tested on GTX 1080)
- Python 3.x with matplotlib, pandas, numpy (for plotting)
- ImageMagick (optional, for image resizing)
- curl or wget (for downloading test images)

## Build

```bash
# Build everything
make all

# Build only CPU version
make cpu

# Build only CUDA version  
make cuda

# Clean build files
make clean
```

## Test Images Setup

Before running benchmarks with real images, download test datasets:

```bash
# Download all datasets (Kodak + standard images + synthetic)
./scripts/setup_images.sh --all

# Or download specific datasets:
./scripts/setup_images.sh --kodak      # Kodak dataset (24 images, ~4MB)
./scripts/setup_images.sh --standard   # Standard test images (Lena, Baboon, etc.)
./scripts/setup_images.sh --synthetic  # Generate synthetic test images
```

### Available Datasets

| Dataset | Images | Resolution | Size | Description |
|---------|--------|------------|------|-------------|
| Kodak | 24 | 768×512 | ~4MB | Industry standard for image processing |
| Standard | 8 | 256-512 | ~2MB | Classic images (Lena, Baboon, Peppers) |
| Synthetic | 25 | 256-4096 | ~10MB | Generated patterns (gradient, checker, plasma) |
| Resized | Variable | 256-4096 | ~20MB | Real images at benchmark sizes |

**Note:** The benchmark system can also generate synthetic images at runtime, so downloading is optional but recommended for realistic results.

## Usage

### Single Image Processing
```bash
# Apply Gaussian blur with CUDA
./bin/imgproc -i images/input/test.png -o images/output/blur.png -k gaussian -s 3 -m cuda

# Apply Sobel edge detection with CPU
./bin/imgproc -i images/input/test.png -o images/output/edges.png -k sobel_x -m cpu

# Compare CPU vs CUDA
./bin/imgproc -i images/input/test.png -o images/output/result.png -k sharpen -m both
```

### Benchmarking
```bash
# Run comprehensive benchmarks
./bin/imgproc --benchmark

# Run benchmarks with custom output
./bin/imgproc --benchmark --output results/benchmarks/

# Generate plots from benchmark results
python3 scripts/generate_plots.py
```

### Full Benchmark Pipeline
```bash
# Run everything: build, benchmark, generate plots
./scripts/benchmark.sh
```

### Command Line Options
| Option | Description |
|--------|-------------|
| `-i, --input` | Input image path |
| `-o, --output` | Output image path |
| `-k, --kernel` | Kernel type: `box`, `gaussian`, `sharpen`, `sobel_x`, `sobel_y` |
| `-s, --size` | Kernel size: `3`, `5`, `7` |
| `-m, --mode` | Execution mode: `cpu`, `cuda`, `both` |
| `-b, --blocksize` | CUDA block size: `8`, `16`, `32` (default: 16) |
| `--benchmark` | Run comprehensive benchmarks |
| `--output` | Benchmark output directory |
| `-v, --verbose` | Verbose output |
| `-h, --help` | Show help |

## Project Structure

```
kernel-image-processing/
├── README.md
├── Makefile
├── .gitignore
├── src/
│   ├── main.cpp                # Entry point, CLI
│   ├── image_io.cpp/h          # Image loading/saving
│   ├── kernels.cpp/h           # Convolution kernel definitions
│   ├── cpu_convolution.cpp/h   # Sequential CPU implementation
│   ├── gpu_convolution.cu/cuh  # CUDA implementations
│   ├── benchmark.cpp/h         # Benchmarking system
│   └── utils.cpp/h             # Utilities
├── include/
│   ├── stb_image.h             # Image loading (header-only)
│   └── stb_image_write.h       # Image writing (header-only)
├── images/
│   ├── input/                  # Test images
│   └── output/                 # Processed images
├── results/
│   ├── benchmarks/             # CSV benchmark results
│   └── plots/                  # Generated plots
├── scripts/
│   ├── benchmark.sh            # Full benchmark script
│   ├── generate_plots.py       # Plot generation
│   └── download_images.sh      # Download test images
├── docs/
│   ├── report.pdf              # Technical report
│   └── presentation.pdf        # Presentation slides
└── tests/
    └── test_correctness.cpp    # Correctness validation
```

## Results Summary

### Best Achieved Speedup
| Image Size | Kernel | CPU Time | CUDA Time | Speedup |
|------------|--------|----------|-----------|---------|
| 4096x4096 | Gaussian 5x5 | ~3200 ms | ~18 ms | **~178x** |

### Key Findings
- Speedup increases with image size (more parallelism)
- Speedup increases with kernel size (more computation per pixel)
- Shared memory optimization provides ~1.5x improvement over global memory
- Optimal block size: 16x16 for most cases

See `docs/report.pdf` for detailed analysis.

## Performance Analysis

The project includes comprehensive benchmarks analyzing:
- Speedup vs Image Size (256x256 to 4096x4096)
- Speedup vs Kernel Size (3x3, 5x5, 7x7)
- Impact of CUDA block size (8x8, 16x16, 32x32)
- Incremental optimization gains (global → constant → shared memory)

## References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Image Convolution - Wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing))
- [stb_image library](https://github.com/nothings/stb)

## License

MIT License - See LICENSE file for details.
