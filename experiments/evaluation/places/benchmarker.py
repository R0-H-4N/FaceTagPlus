import torch
import time
import psutil
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import gc


def benchmark_batch_sizes(models_dict, dataset, batch_sizes=None, device='cpu', 
                          num_iterations=30, warmup=3, save_dir='./benchmark_results',
                          sample_subset=None):
    """
    Optimized benchmark for multiple models across different batch sizes
    
    Args:
        models_dict: Dictionary of {model_name: model}
        dataset: PyTorch Dataset to use for benchmarking
        batch_sizes: List of batch sizes to test (default: [1, 4, 8, 16, 32])
        device: 'cpu' or 'cuda'
        num_iterations: Number of batches to process (default: 30)
        warmup: Number of warmup iterations (default: 3)
        save_dir: Directory to save results
        sample_subset: If set, use only this many samples from dataset (speeds up testing)
        
    Returns:
        DataFrame with all benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]  # Reduced default batch sizes
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Use subset of dataset if specified
    if sample_subset and len(dataset) > sample_subset:
        dataset = torch.utils.data.Subset(dataset, range(sample_subset))
        print(f"Using {sample_subset} samples from dataset for faster benchmarking")
    
    process = psutil.Process(os.getpid())
    all_results = []
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*80}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*80}")
        
        model = model.to(device)
        model.eval()
        
        # Get model size info (computed once per model)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        num_params = sum(p.numel() for p in model.parameters())
        
        for batch_size in tqdm(batch_sizes, desc=f"{model_name} batch sizes"):
            # Create dataloader for this batch size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,  # shuffle=False for consistency
                                   num_workers=0, pin_memory=(device=='cuda'), 
                                   drop_last=True)  # drop_last ensures consistent batch sizes
            
            # Get baseline memory
            if device == 'cpu':
                gc.collect()  # Clean memory before measuring
                baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Warmup phase (reduced iterations)
            dataloader_iter = iter(dataloader)
            with torch.no_grad():
                for _ in range(min(warmup, len(dataloader))):
                    try:
                        images, _ = next(dataloader_iter)
                        images = images.to(device, non_blocking=True)
                        _ = model(images)
                        if device == 'cuda':
                            torch.cuda.synchronize()
                    except StopIteration:
                        break
            
            # Clear cache after warmup
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Benchmark phase
            times = []
            cpu_usages = []
            memory_usages = []
            
            dataloader_iter = iter(dataloader)
            actual_iterations = min(num_iterations, len(dataloader))
            
            with torch.no_grad():
                for i in range(actual_iterations):
                    try:
                        images, _ = next(dataloader_iter)
                    except StopIteration:
                        break
                    
                    actual_batch_size = images.size(0)
                    images = images.to(device, non_blocking=True)
                    
                    # Measure inference time
                    if device == 'cuda':
                        torch.cuda.synchronize()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        _ = model(images)
                        end.record()
                        torch.cuda.synchronize()
                        batch_time_ms = start.elapsed_time(end)
                    else:
                        start = time.perf_counter()
                        _ = model(images)
                        end = time.perf_counter()
                        batch_time_ms = (end - start) * 1000
                    
                    per_image_time_ms = batch_time_ms / actual_batch_size
                    times.append(per_image_time_ms)
                    
                    # Measure CPU and memory (only on CPU, and sample every 5th iteration)
                    if device == 'cpu' and i % 5 == 0:
                        cpu_usages.append(process.cpu_percent(interval=None))
                        memory_usages.append(process.memory_info().rss / 1024 / 1024)
            
            # Clear cache after benchmark
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Compute statistics using percentiles for robustness
            times_array = np.array(times)
            result = {
                'model': model_name,
                'batch_size': batch_size,
                'device': device,
                'model_size_mb': model_size_mb,
                'num_parameters_millions': num_params / 1e6,
                'iterations_completed': len(times),
                
                # Time metrics (using percentiles to filter outliers)
                'mean_time_per_image_ms': np.mean(times_array),
                'std_time_per_image_ms': np.std(times_array),
                'min_time_per_image_ms': np.percentile(times_array, 5),  # 5th percentile
                'max_time_per_image_ms': np.percentile(times_array, 95),  # 95th percentile
                'median_time_per_image_ms': np.median(times_array),
                
                # Throughput
                'throughput_images_per_sec': 1000.0 / np.mean(times_array),
                'batch_throughput_batches_per_sec': 1000.0 / (np.mean(times_array) * batch_size),
            }
            
            # Add CPU/memory metrics for CPU device
            if device == 'cpu' and cpu_usages:
                result.update({
                    'mean_cpu_percent': np.mean(cpu_usages),
                    'max_cpu_percent': np.max(cpu_usages),
                    'mean_memory_mb': np.mean(memory_usages) if memory_usages else 0,
                    'max_memory_mb': np.max(memory_usages) if memory_usages else 0,
                    'memory_increase_mb': (np.mean(memory_usages) - baseline_memory) if memory_usages else 0,
                })
            
            all_results.append(result)
        
        # Clean up after each model
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    csv_path = os.path.join(save_dir, f'benchmark_batch_sizes_{device}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Generate plots
    plot_benchmark_results(results_df, save_dir, device)
    
    return results_df


def plot_benchmark_results(results_df, save_dir, device):
    """
    Create optimized visualization plots for benchmark results
    """
    models = results_df['model'].unique()
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100  # Reduced DPI for faster rendering
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model Benchmark Comparison ({device.upper()})', fontsize=16, y=0.995)
    
    # Plot 1: Time per image vs batch size
    ax = axes[0, 0]
    for model in models:
        model_data = results_df[results_df['model'] == model]
        ax.plot(model_data['batch_size'], model_data['mean_time_per_image_ms'], 
               marker='o', label=model, linewidth=2, markersize=6)
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Time per Image (ms)', fontsize=11)
    ax.set_title('Inference Time vs Batch Size', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Plot 2: Throughput vs batch size
    ax = axes[0, 1]
    for model in models:
        model_data = results_df[results_df['model'] == model]
        ax.plot(model_data['batch_size'], model_data['throughput_images_per_sec'], 
               marker='s', label=model, linewidth=2, markersize=6)
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Throughput (images/sec)', fontsize=11)
    ax.set_title('Throughput vs Batch Size', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Plot 3: CPU usage (if available)
    ax = axes[1, 0]
    if 'mean_cpu_percent' in results_df.columns:
        for model in models:
            model_data = results_df[results_df['model'] == model]
            ax.plot(model_data['batch_size'], model_data['mean_cpu_percent'], 
                   marker='^', label=model, linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size', fontsize=11)
        ax.set_ylabel('CPU Usage (%)', fontsize=11)
        ax.set_title('CPU Usage vs Batch Size', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    else:
        ax.text(0.5, 0.5, 'CPU metrics not available\n(GPU mode)', 
               ha='center', va='center', fontsize=11)
        ax.axis('off')
    
    # Plot 4: Memory usage (if available)
    ax = axes[1, 1]
    if 'mean_memory_mb' in results_df.columns:
        for model in models:
            model_data = results_df[results_df['model'] == model]
            ax.plot(model_data['batch_size'], model_data['mean_memory_mb'], 
                   marker='d', label=model, linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size', fontsize=11)
        ax.set_ylabel('Memory Usage (MB)', fontsize=11)
        ax.set_title('Memory Usage vs Batch Size', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    else:
        ax.text(0.5, 0.5, 'Memory metrics not available\n(GPU mode)', 
               ha='center', va='center', fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'benchmark_plots_{device}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')  # Reduced DPI
    print(f"📊 Plots saved to: {plot_path}")
    plt.close()
    
    # Create comparison table plot
    create_comparison_table(results_df, save_dir, device)


def create_comparison_table(results_df, save_dir, device):
    """
    Create a summary comparison table
    """
    # Get data for batch_size=1 (single image inference)
    single_image_df = results_df[results_df['batch_size'] == 1].copy()
    
    if len(single_image_df) == 0:
        # Use smallest batch size available
        min_batch = results_df['batch_size'].min()
        single_image_df = results_df[results_df['batch_size'] == min_batch].copy()
    
    fig, ax = plt.subplots(figsize=(12, 3 + len(single_image_df) * 0.4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Model', 'Size (MB)', 'Params (M)', 'Time (ms)', 'FPS', 'Throughput (img/s)']
    
    if 'mean_cpu_percent' in single_image_df.columns:
        headers.extend(['CPU (%)', 'RAM (MB)'])
    
    for _, row in single_image_df.iterrows():
        row_data = [
            row['model'],
            f"{row['model_size_mb']:.1f}",
            f"{row['num_parameters_millions']:.2f}",
            f"{row['mean_time_per_image_ms']:.2f} ± {row['std_time_per_image_ms']:.2f}",
            f"{1000/row['mean_time_per_image_ms']:.1f}",
            f"{row['throughput_images_per_sec']:.1f}"
        ]
        
        if 'mean_cpu_percent' in row:
            row_data.extend([
                f"{row['mean_cpu_percent']:.1f}",
                f"{row['mean_memory_mb']:.1f}"
            ])
        
        table_data.append(row_data)
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    batch_label = single_image_df['batch_size'].iloc[0]
    plt.title(f'Model Comparison Summary (Batch Size={batch_label}, {device.upper()})', 
             fontsize=13, weight='bold', pad=15)
    
    table_path = os.path.join(save_dir, f'comparison_table_{device}.png')
    plt.savefig(table_path, dpi=150, bbox_inches='tight')  # Reduced DPI
    print(f"📋 Comparison table saved to: {table_path}")
    plt.close()


# Optimized usage example
