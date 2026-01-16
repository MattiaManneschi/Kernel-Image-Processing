# ============================================================================
# Kernel Image Processing - Makefile
# ============================================================================

# Compiler settings
CXX := g++
NVCC := nvcc

# Flags
CXXFLAGS := -O3 -Wall -Wextra -std=c++17 -fopenmp
NVCCFLAGS := -O3 -std=c++17 -arch=sm_61 -Xcompiler -fopenmp -ccbin=/usr/bin/g++-12
# sm_61 = GTX 1080, change if using different GPU:
# sm_75 = RTX 2080, sm_86 = RTX 3080, sm_89 = RTX 4080

# Debug flags (use: make DEBUG=1)
ifdef DEBUG
    CXXFLAGS += -g -DDEBUG
    NVCCFLAGS += -g -G -DDEBUG
endif

# Directories
SRC_DIR := src
INC_DIR := include
BIN_DIR := bin
OBJ_DIR := obj

# Source files
CPP_SRCS := $(SRC_DIR)/main.cpp \
            $(SRC_DIR)/image_io.cpp \
            $(SRC_DIR)/kernels.cpp \
            $(SRC_DIR)/cpu_convolution.cpp \
            $(SRC_DIR)/benchmark.cpp \
            $(SRC_DIR)/utils.cpp

CU_SRCS := $(SRC_DIR)/gpu_convolution.cu

# Object files
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_SRCS))

# All objects
ALL_OBJS := $(CPP_OBJS) $(CU_OBJS)

# Target executable
TARGET := $(BIN_DIR)/imgproc

# Include paths
INCLUDES := -I$(INC_DIR) -I$(SRC_DIR)

# CUDA libraries
LDFLAGS := -lcudart -lm

# ============================================================================
# Rules
# ============================================================================

.PHONY: all clean directories cpu cuda benchmark plots help

# Default target
all: directories $(TARGET)
	@echo "Build complete: $(TARGET)"

# Create directories
directories:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)
	@mkdir -p images/input images/output
	@mkdir -p results/benchmarks results/plots

# Link final executable
$(TARGET): $(ALL_OBJS)
	@echo "Linking $@..."
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# Compile C++ sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA sources
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# CPU-only build (no CUDA)
cpu: CXXFLAGS += -DNO_CUDA
cpu: directories $(CPP_OBJS)
	@echo "Building CPU-only version..."
	$(CXX) $(CXXFLAGS) $(CPP_OBJS) -o $(BIN_DIR)/imgproc_cpu -lm

# Run benchmarks
benchmark: all
	@echo "Running benchmarks..."
	./$(TARGET) --benchmark --output results/benchmarks/
	@echo "Benchmarks complete. Results in results/benchmarks/"

# Generate plots
plots:
	@echo "Generating plots..."
	python3 scripts/generate_plots.py
	@echo "Plots saved to results/plots/"

# Full pipeline
full: benchmark plots
	@echo "Full benchmark pipeline complete!"

# Clean build files
clean:
	@echo "Cleaning..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete."

# Deep clean (including results)
distclean: clean
	rm -rf results/benchmarks/*.csv
	rm -rf results/plots/*.png
	rm -rf images/output/*

# Help
help:
	@echo "Kernel Image Processing - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build complete project (default)"
	@echo "  cpu        - Build CPU-only version"
	@echo "  benchmark  - Build and run benchmarks"
	@echo "  plots      - Generate plots from benchmark results"
	@echo "  full       - Run complete pipeline (benchmark + plots)"
	@echo "  clean      - Remove build files"
	@echo "  distclean  - Remove build files and results"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1    - Build with debug symbols"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build project"
	@echo "  make DEBUG=1            # Build with debug"
	@echo "  make benchmark          # Build and run benchmarks"
	@echo "  make full               # Complete pipeline"

# Dependencies (auto-generated would be better, but this works)
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/image_io.h $(SRC_DIR)/kernels.h \
                   $(SRC_DIR)/cpu_convolution.h $(SRC_DIR)/gpu_convolution.cuh \
                   $(SRC_DIR)/benchmark.h $(SRC_DIR)/utils.h
$(OBJ_DIR)/image_io.o: $(SRC_DIR)/image_io.cpp $(SRC_DIR)/image_io.h
$(OBJ_DIR)/kernels.o: $(SRC_DIR)/kernels.cpp $(SRC_DIR)/kernels.h
$(OBJ_DIR)/cpu_convolution.o: $(SRC_DIR)/cpu_convolution.cpp $(SRC_DIR)/cpu_convolution.h
$(OBJ_DIR)/benchmark.o: $(SRC_DIR)/benchmark.cpp $(SRC_DIR)/benchmark.h
$(OBJ_DIR)/utils.o: $(SRC_DIR)/utils.cpp $(SRC_DIR)/utils.h
$(OBJ_DIR)/gpu_convolution.cu.o: $(SRC_DIR)/gpu_convolution.cu $(SRC_DIR)/gpu_convolution.cuh
