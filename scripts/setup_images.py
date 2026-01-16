#!/usr/bin/env python3
"""
Download and prepare test images for Kernel Image Processing benchmarks.

Requirements:
    pip install kagglehub pillow

Usage:
    python setup_images.py [--kodak] [--synthetic] [--all]
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Project directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
IMAGE_DIR = PROJECT_DIR / "images" / "input"
OUTPUT_DIR = PROJECT_DIR / "images" / "output"

# Benchmark image sizes
SIZES = [256, 512, 1024, 2048, 4096]


def setup_directories():
    """Create necessary directories."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (IMAGE_DIR / "kodak").mkdir(exist_ok=True)
    (IMAGE_DIR / "synthetic").mkdir(exist_ok=True)
    (IMAGE_DIR / "resized").mkdir(exist_ok=True)


def download_kodak():
    """Download Kodak dataset from Kaggle."""
    print("\n=== Downloading Kodak Dataset ===\n")

    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed")
        print("Run: pip install kagglehub")
        return False

    try:
        # Download dataset
        print("Downloading from Kaggle...")
        path = kagglehub.dataset_download("sherylmehta/kodak-dataset")
        print(f"Downloaded to: {path}")

        # Copy to project directory
        kodak_dir = IMAGE_DIR / "kodak"
        source_path = Path(path)

        # Find PNG files (might be in subdirectory)
        png_files = list(source_path.rglob("*.png"))

        if not png_files:
            print("ERROR: No PNG files found in dataset")
            return False

        print(f"Copying {len(png_files)} images to {kodak_dir}")
        for png in png_files:
            dest = kodak_dir / png.name
            shutil.copy2(png, dest)
            print(f"  Copied: {png.name}")

        print(f"\n✓ Kodak dataset ready: {len(png_files)} images")
        return True

    except Exception as e:
        print(f"ERROR: Failed to download Kodak dataset: {e}")
        return False


def generate_synthetic():
    """Generate synthetic test images at various sizes."""
    print("\n=== Generating Synthetic Images ===\n")

    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except ImportError:
        print("ERROR: Pillow not installed")
        print("Run: pip install pillow numpy")
        return False

    synth_dir = IMAGE_DIR / "synthetic"
    count = 0

    for size in SIZES:
        print(f"Generating {size}x{size} images...")

        # Gradient (red to blue diagonal)
        gradient_path = synth_dir / f"gradient_{size}.png"
        if not gradient_path.exists():
            img = np.zeros((size, size, 3), dtype=np.uint8)
            for y in range(size):
                for x in range(size):
                    img[y, x, 0] = int(255 * x / size)  # Red
                    img[y, x, 2] = int(255 * y / size)  # Blue
            Image.fromarray(img).save(gradient_path)
            count += 1

        # Checkerboard
        checker_path = synth_dir / f"checker_{size}.png"
        if not checker_path.exists():
            img = Image.new('RGB', (size, size), 'white')
            draw = ImageDraw.Draw(img)
            block = size // 8
            for y in range(0, size, block):
                for x in range(0, size, block):
                    if (x // block + y // block) % 2 == 0:
                        draw.rectangle([x, y, x + block, y + block], fill='black')
            img.save(checker_path)
            count += 1

        # Noise
        noise_path = synth_dir / f"noise_{size}.png"
        if not noise_path.exists():
            img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
            Image.fromarray(img).save(noise_path)
            count += 1

        # Circles (good for blur testing)
        circles_path = synth_dir / f"circles_{size}.png"
        if not circles_path.exists():
            img = Image.new('RGB', (size, size), 'white')
            draw = ImageDraw.Draw(img)
            center = size // 2
            for r in range(size // 10, size // 2, size // 10):
                draw.ellipse([center - r, center - r, center + r, center + r],
                             outline='black', width=2)
            img.save(circles_path)
            count += 1

        # Solid gray
        gray_path = synth_dir / f"gray_{size}.png"
        if not gray_path.exists():
            img = Image.new('RGB', (size, size), (128, 128, 128))
            img.save(gray_path)
            count += 1

    print(f"\n✓ Generated {count} synthetic images")
    return True


def resize_kodak():
    """Resize Kodak images to benchmark sizes."""
    print("\n=== Resizing Kodak Images ===\n")

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed")
        print("Run: pip install pillow")
        return False

    kodak_dir = IMAGE_DIR / "kodak"
    resized_dir = IMAGE_DIR / "resized"

    # Get first 5 Kodak images for resizing
    kodak_images = sorted(kodak_dir.glob("*.png"))[:5]

    if not kodak_images:
        print("No Kodak images found. Run with --kodak first.")
        return False

    count = 0
    for img_path in kodak_images:
        img = Image.open(img_path)
        name = img_path.stem

        for size in SIZES:
            output_path = resized_dir / f"{name}_{size}x{size}.png"
            if not output_path.exists():
                resized = img.resize((size, size), Image.LANCZOS)
                resized.save(output_path)
                print(f"  Created: {output_path.name}")
                count += 1

    print(f"\n✓ Resized {count} images")
    return True


def print_summary():
    """Print summary of available images."""
    print("\n" + "=" * 50)
    print("Image Setup Complete")
    print("=" * 50 + "\n")

    print(f"Image directory: {IMAGE_DIR}\n")
    print("Contents:")

    kodak_dir = IMAGE_DIR / "kodak"
    if kodak_dir.exists():
        kodak_count = len(list(kodak_dir.glob("*.png")))
        print(f"  - Kodak: {kodak_count} images")

    synth_dir = IMAGE_DIR / "synthetic"
    if synth_dir.exists():
        synth_count = len(list(synth_dir.glob("*.png")))
        print(f"  - Synthetic: {synth_count} images")

    resized_dir = IMAGE_DIR / "resized"
    if resized_dir.exists():
        resized_count = len(list(resized_dir.glob("*.png")))
        print(f"  - Resized: {resized_count} images")

    total = len(list(IMAGE_DIR.rglob("*.png")))
    print(f"\nTotal: {total} images")

    print("\nReady for benchmarking! Run:")
    print("  ./bin/imgproc --benchmark")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare test images for benchmarking"
    )
    parser.add_argument("--kodak", action="store_true",
                        help="Download Kodak dataset from Kaggle")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic test images")
    parser.add_argument("--resize", action="store_true",
                        help="Resize Kodak images to benchmark sizes")
    parser.add_argument("--all", action="store_true",
                        help="Download and generate all images")

    args = parser.parse_args()

    # Default to --all if no arguments
    if not any([args.kodak, args.synthetic, args.resize, args.all]):
        args.all = True

    print("\n" + "=" * 50)
    print("Kernel Image Processing - Image Setup")
    print("=" * 50)

    setup_directories()

    if args.all or args.kodak:
        download_kodak()

    if args.all or args.synthetic:
        generate_synthetic()

    if args.all or args.resize:
        resize_kodak()

    print_summary()


if __name__ == "__main__":
    main()