cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
project(KernelImageProcessing LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()


find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)


if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

    set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)
endif ()

message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")


find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found and enabled")
endif ()


set(CPP_SOURCES
        src/main.cpp
        src/image_io.cpp
        src/kernels.cpp
        src/cpu_convolution.cpp
        src/benchmark.cpp
        src/advanced_tests.cpp
        src/utils.cpp
)

set(CUDA_SOURCES
        src/gpu_convolution.cu
)

set(HEADERS
        src/image_io.h
        src/kernels.h
        src/cpu_convolution.h
        src/gpu_convolution.cuh
        src/benchmark.h
        src/advanced_tests.h
        src/utils.h
)


include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)


add_executable(imgproc
        ${CPP_SOURCES}
        ${CUDA_SOURCES}
        ${HEADERS}
)

target_link_libraries(imgproc
        CUDA::cudart
        ${CMAKE_DL_LIBS}
)

if (OpenMP_CXX_FOUND)
    target_link_libraries(imgproc OpenMP::OpenMP_CXX)
endif ()


set_target_properties(imgproc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)


option(BUILD_CPU_ONLY "Build CPU-only version without CUDA" OFF)

if (BUILD_CPU_ONLY)
    add_executable(imgproc_cpu
            ${CPP_SOURCES}
            ${HEADERS}
    )

    target_compile_definitions(imgproc_cpu PRIVATE NO_CUDA)

    if (OpenMP_CXX_FOUND)
        target_link_libraries(imgproc_cpu OpenMP::OpenMP_CXX)
    endif ()

    set_target_properties(imgproc_cpu PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    )
endif ()


file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/images/input)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/images/input/kodak)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/images/output)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results/benchmarks)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results/plots)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results/advanced_tests)


install(TARGETS imgproc
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
)


add_custom_target(benchmark
        COMMAND ${CMAKE_SOURCE_DIR}/bin/imgproc --benchmark
        DEPENDS imgproc
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running benchmarks..."
)


add_custom_target(advanced_tests
        COMMAND ${CMAKE_SOURCE_DIR}/bin/imgproc --advanced-tests --images-dir images/input/kodak
        DEPENDS imgproc
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running advanced tests..."
)


add_custom_target(generate_examples
        COMMAND ${CMAKE_SOURCE_DIR}/bin/imgproc -i images/input/kodak/kodim01.png --generate-all
        DEPENDS imgproc
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating filtered images..."
)


add_custom_target(download_kodak
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/download_images.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Downloading Kodak test images..."
)


message(STATUS "")
message(STATUS "=== Kernel Image Processing Configuration ===")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler:     ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version:      ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenMP:            ${OpenMP_CXX_FOUND}")
message(STATUS "CPU-only build:    ${BUILD_CPU_ONLY}")
message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  cmake --build . --target imgproc")
message(STATUS "  cmake --build . --target benchmark")
message(STATUS "  cmake --build . --target advanced_tests")
message(STATUS "  cmake --build . --target generate_examples")