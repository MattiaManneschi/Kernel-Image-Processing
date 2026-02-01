#!/bin/bash
# =============================================================================
# Run Advanced Tests - Linux
# With automatic dependency installation
# =============================================================================

set -e

# Change to script directory
cd "$(dirname "$0")"

echo "=============================================="
echo "  Kernel Image Processing - Advanced Tests"
echo "=============================================="
echo ""

# =============================================================================
# Check and install dependencies
# =============================================================================

install_dependencies() {
    echo "Checking dependencies..."
    echo ""

    MISSING_DEPS=""

    # Check for CUDA
    if ! command -v nvcc &> /dev/null; then
        echo "[!] CUDA toolkit not found"
        MISSING_DEPS="$MISSING_DEPS cuda"
    else
        echo "[OK] CUDA toolkit: $(nvcc --version | grep release | awk '{print $6}')"
    fi

    # Check for g++
    if ! command -v g++ &> /dev/null; then
        echo "[!] g++ not found"
        MISSING_DEPS="$MISSING_DEPS g++"
    else
        echo "[OK] g++: $(g++ --version | head -n1)"
    fi

    # Check for g++-12 (needed for CUDA compatibility)
    if ! command -v g++-12 &> /dev/null; then
        echo "[!] g++-12 not found (required for CUDA)"
        MISSING_DEPS="$MISSING_DEPS g++-12"
    else
        echo "[OK] g++-12: $(g++-12 --version | head -n1)"
    fi

    # Check for make
    if ! command -v make &> /dev/null; then
        echo "[!] make not found"
        MISSING_DEPS="$MISSING_DEPS make"
    else
        echo "[OK] make: $(make --version | head -n1)"
    fi

    # Check for Python3
    if ! command -v python3 &> /dev/null; then
        echo "[!] python3 not found"
        MISSING_DEPS="$MISSING_DEPS python3"
    else
        echo "[OK] python3: $(python3 --version)"
    fi

    # Check for curl or wget
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        echo "[!] curl/wget not found"
        MISSING_DEPS="$MISSING_DEPS curl"
    else
        echo "[OK] Download tool: $(command -v curl || command -v wget)"
    fi

    echo ""

    # Install missing dependencies
    if [ -n "$MISSING_DEPS" ]; then
        echo "Missing dependencies:$MISSING_DEPS"
        echo ""

        # Detect package manager
        if command -v apt-get &> /dev/null; then
            PKG_MANAGER="apt"
        elif command -v dnf &> /dev/null; then
            PKG_MANAGER="dnf"
        elif command -v pacman &> /dev/null; then
            PKG_MANAGER="pacman"
        else
            echo "ERROR: Could not detect package manager (apt/dnf/pacman)"
            echo "Please install manually:$MISSING_DEPS"
            exit 1
        fi

        echo "Installing missing dependencies..."
        echo "(This may require sudo password)"
        echo ""

        case $PKG_MANAGER in
            apt)
                sudo apt-get update
                for dep in $MISSING_DEPS; do
                    case $dep in
                        cuda)
                            echo ""
                            echo "=========================================="
                            echo "CUDA must be installed manually!"
                            echo "=========================================="
                            echo "Visit: https://developer.nvidia.com/cuda-downloads"
                            echo ""
                            echo "Or on Ubuntu/Debian:"
                            echo "  sudo apt install nvidia-cuda-toolkit"
                            echo ""
                            read -p "Press Enter after installing CUDA..."
                            ;;
                        g++)
                            sudo apt-get install -y build-essential
                            ;;
                        g++-12)
                            sudo apt-get install -y gcc-12 g++-12
                            ;;
                        make)
                            sudo apt-get install -y build-essential
                            ;;
                        python3)
                            sudo apt-get install -y python3 python3-pip
                            ;;
                        curl)
                            sudo apt-get install -y curl
                            ;;
                    esac
                done
                ;;
            dnf)
                for dep in $MISSING_DEPS; do
                    case $dep in
                        cuda)
                            echo "CUDA must be installed manually from NVIDIA website:"
                            echo "https://developer.nvidia.com/cuda-downloads"
                            read -p "Press Enter after installing CUDA..."
                            ;;
                        g++|make)
                            sudo dnf install -y gcc-c++ make
                            ;;
                        g++-12)
                            sudo dnf install -y gcc-toolset-12-gcc-c++
                            ;;
                        python3)
                            sudo dnf install -y python3 python3-pip
                            ;;
                        curl)
                            sudo dnf install -y curl
                            ;;
                    esac
                done
                ;;
            pacman)
                for dep in $MISSING_DEPS; do
                    case $dep in
                        cuda)
                            sudo pacman -S --noconfirm cuda
                            ;;
                        g++|make)
                            sudo pacman -S --noconfirm base-devel
                            ;;
                        g++-12)
                            echo "On Arch, default gcc should work."
                            ;;
                        python3)
                            sudo pacman -S --noconfirm python python-pip
                            ;;
                        curl)
                            sudo pacman -S --noconfirm curl
                            ;;
                    esac
                done
                ;;
        esac

        echo ""
        echo "Dependencies installed."
        echo ""
    fi
}

# =============================================================================
# Install Python packages for plotting
# =============================================================================

install_python_packages() {
    echo "Checking Python packages..."

    MISSING_PKGS=""

    python3 -c "import matplotlib" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS matplotlib"
    python3 -c "import pandas" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS pandas"
    python3 -c "import numpy" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS numpy"

    if [ -n "$MISSING_PKGS" ]; then
        echo "Installing Python packages:$MISSING_PKGS"
        pip3 install --user --break-system-packages $MISSING_PKGS 2>/dev/null || \
        pip3 install --user $MISSING_PKGS 2>/dev/null || \
        pip install --user $MISSING_PKGS
        echo ""
    else
        echo "[OK] Python packages: matplotlib, pandas, numpy"
    fi
    echo ""
}

# =============================================================================
# Download Kodak images if missing
# =============================================================================

download_kodak_images() {
    KODAK_DIR="./images/input/kodak"

    # Check if images exist
    if [ -d "$KODAK_DIR" ] && [ "$(ls -A $KODAK_DIR/*.png 2>/dev/null | wc -l)" -ge 20 ]; then
        echo "[OK] Kodak images found in $KODAK_DIR"
        echo ""
        return
    fi

    echo "Kodak images not found or incomplete. Downloading..."
    mkdir -p "$KODAK_DIR"

    for i in $(seq 1 24); do
        NUM=$(printf "%02d" $i)
        FILE="$KODAK_DIR/kodim${NUM}.png"

        if [ -f "$FILE" ]; then
            continue
        fi

        URL="http://r0k.us/graphics/kodak/kodak/kodim${NUM}.png"
        echo "  Downloading kodim${NUM}.png..."

        if command -v curl &> /dev/null; then
            curl -s -f -o "$FILE" "$URL" || echo "    Failed: $URL"
        elif command -v wget &> /dev/null; then
            wget -q -O "$FILE" "$URL" || echo "    Failed: $URL"
        fi
    done

    echo ""
    echo "Kodak images downloaded."
    echo ""
}

# =============================================================================
# Build project if needed
# =============================================================================

build_project() {
    if [ ! -f "./bin/imgproc" ]; then
        echo "Executable not found. Building project..."
        echo ""
        make clean 2>/dev/null || true
        make all
        echo ""
        echo "Build complete."
        echo ""
    else
        echo "[OK] Executable found: ./bin/imgproc"

        # Check if source is newer than binary
        if [ -n "$(find ./src -name '*.cpp' -newer ./bin/imgproc 2>/dev/null)" ] || \
           [ -n "$(find ./src -name '*.cu' -newer ./bin/imgproc 2>/dev/null)" ]; then
            echo "    Source files changed, rebuilding..."
            make all
            echo ""
        fi
        echo ""
    fi
}

# =============================================================================
# Main
# =============================================================================

# Run checks and installations
install_dependencies
install_python_packages
download_kodak_images
build_project

# Run the tests
echo "=============================================="
echo "  Running Advanced Tests"
echo "=============================================="
echo ""

if [ -d "./images/input/kodak" ] && [ -n "$(ls -A ./images/input/kodak/*.png 2>/dev/null)" ]; then
    ./bin/imgproc --advanced-tests --images-dir images/input/kodak
else
    echo "WARNING: Running with synthetic images"
    ./bin/imgproc --advanced-tests
fi

echo ""
echo "=============================================="
echo "  Tests Complete!"
echo "=============================================="
echo ""
echo "Results saved to: results/advanced_tests/advanced_results.csv"
echo ""

# Keep terminal open
read -p "Press Enter to exit..."