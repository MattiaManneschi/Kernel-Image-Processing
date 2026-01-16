#!/bin/bash
# ============================================================================
# Kernel Image Processing - Full Benchmark Script
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "Kernel Image Processing - Benchmark Suite"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create output directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p bin obj
mkdir -p images/input images/output
mkdir -p results/benchmarks results/plots

# Build project
echo ""
echo -e "${YELLOW}Building project...${NC}"
make clean
make all

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
echo -e "${GREEN}Build successful!${NC}"

# Show CUDA info
echo ""
echo -e "${YELLOW}CUDA Device Info:${NC}"
./bin/imgproc --info

# Run validation
echo ""
echo -e "${YELLOW}Running validation tests...${NC}"
./bin/imgproc --validate -k gaussian -s 3

# Run benchmarks
echo ""
echo -e "${YELLOW}Running benchmarks (this may take several minutes)...${NC}"
./bin/imgproc --benchmark --output-dir results/benchmarks/

# Generate plots
echo ""
echo -e "${YELLOW}Generating plots...${NC}"
if command -v python3 &> /dev/null; then
    python3 scripts/generate_plots.py
else
    echo -e "${RED}Python3 not found. Skipping plot generation.${NC}"
    echo "Install Python3 and run: python3 scripts/generate_plots.py"
fi

# Summary
echo ""
echo "============================================"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo "============================================"
echo ""
echo "Results:"
echo "  - CSV: results/benchmarks/benchmark_results.csv"
echo "  - Plots: results/plots/"
echo "  - Summary: results/plots/summary.txt"
echo ""
