# Kernel Image Processing - CUDA

Parallel implementation of convolution-based image filters using CUDA.

## Description

This project implements 2D image convolution using CUDA parallel computing, comparing a sequential CPU implementation with three GPU-optimized versions (Global, Constant, and Shared memory). The project achieves up to **350x speedup** over sequential CPU execution.

Implemented filters include: Gaussian Blur, Box Blur, Sharpen, Sobel Edge Detection, Laplacian, Prewitt, and Emboss.

## Quick Start

```bash
# Build
make all

# Run advanced tests
./run_advanced_tests.sh
```