#!/bin/bash
# ============================================================================
# Download test images for benchmarking
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_DIR="$PROJECT_DIR/images/input"

mkdir -p "$IMAGE_DIR"
cd "$IMAGE_DIR"

echo "Downloading test images..."

# Kodak test images (standard image processing test set)
# These are public domain images

KODAK_BASE="http://r0k.us/graphics/kodak/kodak"

# Download a few Kodak images
for i in 01 02 03 04 05; do
    if [ ! -f "kodim${i}.png" ]; then
        echo "Downloading kodim${i}.png..."
        curl -s -o "kodim${i}.png" "${KODAK_BASE}/kodim${i}.png" || \
        wget -q -O "kodim${i}.png" "${KODAK_BASE}/kodim${i}.png" || \
        echo "Failed to download kodim${i}.png"
    fi
done

# Generate synthetic test images at various sizes
echo ""
echo "Generating synthetic test images..."

# Check if ImageMagick is available
if command -v convert &> /dev/null; then
    for size in 256 512 1024 2048 4096; do
        if [ ! -f "gradient_${size}.png" ]; then
            echo "Creating gradient_${size}.png..."
            convert -size ${size}x${size} gradient:red-blue "gradient_${size}.png"
        fi
        
        if [ ! -f "noise_${size}.png" ]; then
            echo "Creating noise_${size}.png..."
            convert -size ${size}x${size} plasma:fractal "noise_${size}.png"
        fi
    done
else
    echo "ImageMagick not found. Synthetic images will be generated at runtime."
    echo "Install ImageMagick for pre-generated test images:"
    echo "  Ubuntu/Debian: sudo apt-get install imagemagick"
    echo "  macOS: brew install imagemagick"
fi

echo ""
echo "Test images available in: $IMAGE_DIR"
ls -la "$IMAGE_DIR"
