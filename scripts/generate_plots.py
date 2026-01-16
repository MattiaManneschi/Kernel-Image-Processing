#!/usr/bin/env python3
"""
Generate benchmark plots from CSV results.
Requires: matplotlib, pandas, numpy
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'cpu': '#e74c3c',
    'cuda_global': '#3498db',
    'cuda_constant': '#2ecc71',
    'cuda_shared': '#9b59b6'
}

def load_results(csv_path):
    """Load benchmark results from CSV."""
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} benchmark results")
    return df

def plot_speedup_vs_imagesize(df, output_dir):
    """Plot speedup vs image size for different kernel sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to shared memory only (best CUDA implementation)
    cuda_df = df[df['implementation'] == 'cuda_shared']
    
    for k_size in sorted(cuda_df['kernel_size'].unique()):
        subset = cuda_df[cuda_df['kernel_size'] == k_size]
        grouped = subset.groupby('image_size')['speedup'].mean().reset_index()
        
        ax.plot(grouped['image_size'], grouped['speedup'], 
                marker='o', linewidth=2, markersize=8,
                label=f'Kernel {k_size}x{k_size}')
    
    ax.set_xlabel('Image Size (pixels)', fontsize=12)
    ax.set_ylabel('Speedup (CPU time / CUDA time)', fontsize=12)
    ax.set_title('Speedup vs Image Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    
    # Add speedup reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50x')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_vs_imagesize.png'), dpi=150)
    plt.close()
    print("  Created: speedup_vs_imagesize.png")

def plot_speedup_vs_kernelsize(df, output_dir):
    """Plot speedup vs kernel size for different image sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cuda_df = df[df['implementation'] == 'cuda_shared']
    
    for img_size in sorted(cuda_df['image_size'].unique()):
        subset = cuda_df[cuda_df['image_size'] == img_size]
        grouped = subset.groupby('kernel_size')['speedup'].mean().reset_index()
        
        if len(grouped) > 1:
            ax.plot(grouped['kernel_size'], grouped['speedup'], 
                    marker='s', linewidth=2, markersize=8,
                    label=f'{img_size}x{img_size}')
    
    ax.set_xlabel('Kernel Size', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup vs Kernel Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks([3, 5, 7])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_vs_kernelsize.png'), dpi=150)
    plt.close()
    print("  Created: speedup_vs_kernelsize.png")

def plot_implementation_comparison(df, output_dir):
    """Compare different CUDA implementations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get largest image size for comparison
    max_size = df['image_size'].max()
    subset = df[df['image_size'] == max_size]
    
    implementations = ['cpu', 'cuda_global', 'cuda_constant', 'cuda_shared']
    impl_labels = ['CPU', 'CUDA Global', 'CUDA Constant', 'CUDA Shared']
    
    times = []
    for impl in implementations:
        impl_data = subset[subset['implementation'] == impl]
        if len(impl_data) > 0:
            times.append(impl_data['time_ms'].mean())
        else:
            times.append(0)
    
    colors = [COLORS.get(impl, 'gray') for impl in implementations]
    bars = ax.bar(impl_labels, times, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        if time > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                    f'{time:.2f} ms', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title(f'Implementation Comparison ({max_size}x{max_size} image)', 
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'implementation_comparison.png'), dpi=150)
    plt.close()
    print("  Created: implementation_comparison.png")

def plot_speedup_heatmap(df, output_dir):
    """Create heatmap of speedup values."""
    cuda_df = df[df['implementation'] == 'cuda_shared']
    
    pivot = cuda_df.pivot_table(
        values='speedup', 
        index='kernel_size', 
        columns='image_size', 
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    cbar = plt.colorbar(im, label='Speedup')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{k}x{k}' for k in pivot.index])
    
    ax.set_xlabel('Image Size', fontsize=12)
    ax.set_ylabel('Kernel Size', fontsize=12)
    ax.set_title('Speedup Heatmap (CUDA Shared Memory)', fontsize=14, fontweight='bold')
    
    # Add values in cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > pivot.values.max() * 0.6 else 'black'
                ax.text(j, i, f'{value:.1f}x', ha='center', va='center', 
                        fontsize=10, color=text_color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_heatmap.png'), dpi=150)
    plt.close()
    print("  Created: speedup_heatmap.png")

def plot_execution_time_comparison(df, output_dir):
    """Plot execution time comparison (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    implementations = ['cpu', 'cuda_shared']
    labels = {'cpu': 'CPU Sequential', 'cuda_shared': 'CUDA Shared Memory'}
    
    for impl in implementations:
        impl_df = df[df['implementation'] == impl]
        grouped = impl_df.groupby('image_size')['time_ms'].mean().reset_index()
        
        ax.plot(grouped['image_size'], grouped['time_ms'],
                marker='o', linewidth=2, markersize=8,
                color=COLORS.get(impl, 'gray'),
                label=labels.get(impl, impl))
    
    ax.set_xlabel('Image Size (pixels)', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('Execution Time: CPU vs CUDA', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_comparison.png'), dpi=150)
    plt.close()
    print("  Created: execution_time_comparison.png")

def plot_throughput(df, output_dir):
    """Plot throughput (megapixels per second)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cuda_df = df[df['implementation'] == 'cuda_shared']
    
    for k_size in sorted(cuda_df['kernel_size'].unique()):
        subset = cuda_df[cuda_df['kernel_size'] == k_size]
        grouped = subset.groupby('image_size')['throughput_mpps'].mean().reset_index()
        
        ax.plot(grouped['image_size'], grouped['throughput_mpps'],
                marker='o', linewidth=2, markersize=8,
                label=f'Kernel {k_size}x{k_size}')
    
    ax.set_xlabel('Image Size (pixels)', fontsize=12)
    ax.set_ylabel('Throughput (Megapixels/second)', fontsize=12)
    ax.set_title('CUDA Throughput vs Image Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput.png'), dpi=150)
    plt.close()
    print("  Created: throughput.png")

def plot_block_size_comparison(df, output_dir):
    """Compare different CUDA block sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cuda_df = df[df['implementation'] == 'cuda_shared']
    max_size = cuda_df['image_size'].max()
    subset = cuda_df[cuda_df['image_size'] == max_size]
    
    grouped = subset.groupby('block_size')['time_ms'].mean().reset_index()
    
    bars = ax.bar(grouped['block_size'].astype(str), grouped['time_ms'],
                  color=COLORS['cuda_shared'], edgecolor='black', linewidth=1.2)
    
    for bar, time in zip(bars, grouped['time_ms']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time:.2f} ms', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Block Size', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title(f'Block Size Comparison ({max_size}x{max_size} image)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'block_size_comparison.png'), dpi=150)
    plt.close()
    print("  Created: block_size_comparison.png")

def generate_summary_stats(df, output_dir):
    """Generate summary statistics text file."""
    summary_path = os.path.join(output_dir, 'summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BENCHMARK SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Best speedup
        cuda_df = df[df['implementation'].str.startswith('cuda')]
        if len(cuda_df) > 0:
            best = cuda_df.loc[cuda_df['speedup'].idxmax()]
            f.write(f"Best Speedup: {best['speedup']:.2f}x\n")
            f.write(f"  Implementation: {best['implementation']}\n")
            f.write(f"  Image Size: {int(best['image_size'])}x{int(best['image_size'])}\n")
            f.write(f"  Kernel: {best['kernel_name']}\n")
            f.write(f"  Time: {best['time_ms']:.3f} ms\n\n")
        
        # Average speedup by implementation
        f.write("Average Speedup by Implementation:\n")
        for impl in df['implementation'].unique():
            impl_df = df[df['implementation'] == impl]
            avg_speedup = impl_df['speedup'].mean()
            f.write(f"  {impl}: {avg_speedup:.2f}x\n")
        f.write("\n")
        
        # Performance by image size
        f.write("Average Speedup by Image Size (CUDA Shared):\n")
        cuda_shared = df[df['implementation'] == 'cuda_shared']
        for size in sorted(cuda_shared['image_size'].unique()):
            size_df = cuda_shared[cuda_shared['image_size'] == size]
            avg = size_df['speedup'].mean()
            f.write(f"  {int(size)}x{int(size)}: {avg:.2f}x\n")
    
    print(f"  Created: summary.txt")

def main():
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    csv_path = project_dir / 'results' / 'benchmarks' / 'benchmark_results.csv'
    output_dir = project_dir / 'results' / 'plots'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("Generating Benchmark Plots")
    print("=" * 50 + "\n")
    
    # Load data
    df = load_results(csv_path)
    
    # Generate all plots
    print("\nGenerating plots...")
    
    try:
        plot_speedup_vs_imagesize(df, output_dir)
        plot_speedup_vs_kernelsize(df, output_dir)
        plot_implementation_comparison(df, output_dir)
        plot_speedup_heatmap(df, output_dir)
        plot_execution_time_comparison(df, output_dir)
        plot_throughput(df, output_dir)
        plot_block_size_comparison(df, output_dir)
        generate_summary_stats(df, output_dir)
    except Exception as e:
        print(f"Warning: Some plots could not be generated: {e}")
    
    print(f"\nAll plots saved to: {output_dir}")
    print("=" * 50 + "\n")

if __name__ == '__main__':
    main()
