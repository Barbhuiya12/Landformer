"""
Memory Optimization Utilities for VIC Surrogate Model
Author: Siddik
Description: Advanced memory optimization and monitoring utilities
"""

import gc
import os
import psutil
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import time
from functools import wraps


class MemoryMonitor:
    """
    Advanced memory monitoring and optimization
    Author: Siddik
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_history = []
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        memory_info = self.process.memory_info()
        
        usage = {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024,
            'total': psutil.virtual_memory().total / 1024 / 1024
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_memory = torch.cuda.memory_stats(i)
                    usage[f'gpu_{i}_allocated'] = gpu_memory.get('allocated_bytes.all.current', 0) / 1024 / 1024
                    usage[f'gpu_{i}_reserved'] = gpu_memory.get('reserved_bytes.all.current', 0) / 1024 / 1024
                    usage[f'gpu_{i}_max_allocated'] = gpu_memory.get('allocated_bytes.all.peak', 0) / 1024 / 1024
                except Exception as e:
                    # Handle cases where GPU memory stats are not available
                    self.logger.warning(f"Could not get GPU memory stats for device {i}: {e}")
                    usage[f'gpu_{i}_allocated'] = 0
                    usage[f'gpu_{i}_reserved'] = 0
                    usage[f'gpu_{i}_max_allocated'] = 0
        
        return usage
    
    def log_memory_usage(self, tag: str = ""):
        """Log current memory usage"""
        usage = self.get_memory_usage()
        self.memory_history.append((time.time(), tag, usage.copy()))
        
        # Update peak memory
        if usage['rss'] > self.peak_memory['rss']:
            self.peak_memory = usage.copy()
        
        message = f"Memory [{tag}]: RSS={usage['rss']:.1f}MB, "
        message += f"GPU=" + ", ".join([f"{usage.get(f'gpu_{i}_allocated', 0):.1f}MB" 
                                       for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)])
        
        self.logger.debug(message)
    
    def get_memory_summary(self) -> str:
        """Get comprehensive memory summary"""
        current = self.get_memory_usage()
        
        lines = [
            "Memory Summary (by Siddik):",
            "=" * 40,
            f"Current RSS: {current['rss']:.1f} MB",
            f"Peak RSS: {self.peak_memory['rss']:.1f} MB",
            f"Memory usage: {current['percent']:.1f}%",
            f"Available: {current['available']:.1f} MB / {current['total']:.1f} MB"
        ]
        
        if torch.cuda.is_available():
            lines.append("GPU Memory:")
            for i in range(torch.cuda.device_count()):
                allocated = current.get(f'gpu_{i}_allocated', 0)
                reserved = current.get(f'gpu_{i}_reserved', 0)
                max_allocated = current.get(f'gpu_{i}_max_allocated', 0)
                lines.append(f"  GPU {i}: {allocated:.1f} MB allocated, "
                           f"{reserved:.1f} MB reserved, {max_allocated:.1f} MB peak")
        
        return "\n".join(lines)
    
    @contextmanager
    def monitor_block(self, block_name: str):
        """Context manager to monitor memory usage in a code block"""
        self.log_memory_usage(f"{block_name}_start")
        start_memory = self.get_memory_usage()['rss']
        
        try:
            yield self
        finally:
            self.log_memory_usage(f"{block_name}_end")
            end_memory = self.get_memory_usage()['rss']
            delta = end_memory - start_memory
            
            if abs(delta) > 10:  # Only log if significant change
                self.logger.info(f"Memory delta [{block_name}]: {delta:+.1f} MB")


def memory_efficient_function(cleanup_after: bool = True):
    """
    Decorator for memory-efficient function execution
    
    Args:
        cleanup_after: Whether to run garbage collection after function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Clear cache before execution
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                result = func(*args, **kwargs)
            finally:
                if cleanup_after:
                    # Clean up after execution
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            return result
        return wrapper
    return decorator


class ChunkedProcessor:
    """
    Process large datasets in memory-efficient chunks
    Author: Siddik
    """
    
    def __init__(self, max_memory_mb: float = 1000):
        """
        Initialize chunked processor
        
        Args:
            max_memory_mb: Maximum memory to use per chunk in MB
        """
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
    
    def estimate_chunk_size(self, data_shape: tuple, dtype: np.dtype, 
                          safety_factor: float = 0.8) -> int:
        """
        Estimate optimal chunk size based on memory constraints
        
        Args:
            data_shape: Shape of the data array
            dtype: Data type
            safety_factor: Safety factor to avoid memory overflow
            
        Returns:
            Optimal chunk size for first dimension
        """
        # Calculate memory per sample
        sample_size = np.prod(data_shape[1:]) * dtype.itemsize
        samples_per_mb = (1024 * 1024) / sample_size
        
        # Calculate chunk size
        max_chunk_size = int(self.max_memory_mb * samples_per_mb * safety_factor)
        
        # Ensure chunk size is reasonable
        chunk_size = max(1, min(max_chunk_size, data_shape[0]))
        
        self.logger.debug(f"Estimated chunk size: {chunk_size} samples "
                         f"({chunk_size * sample_size / 1024 / 1024:.1f} MB)")
        
        return chunk_size
    
    def process_in_chunks(self, data: np.ndarray, process_func, 
                         chunk_size: Optional[int] = None, **kwargs):
        """
        Process data in memory-efficient chunks
        
        Args:
            data: Input data array
            process_func: Function to apply to each chunk
            chunk_size: Optional chunk size (auto-estimated if None)
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results from processing each chunk
        """
        if chunk_size is None:
            chunk_size = self.estimate_chunk_size(data.shape, data.dtype)
        
        results = []
        n_samples = data.shape[0]
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = data[start_idx:end_idx]
            
            # Process chunk
            try:
                result = process_func(chunk, **kwargs)
                results.append(result)
            finally:
                # Clean up chunk from memory
                del chunk
                gc.collect()
        
        return results


class DataLoaderOptimizer:
    """
    Optimize PyTorch DataLoader parameters for memory efficiency
    Author: Siddik
    """
    
    @staticmethod
    def get_optimal_params(batch_size: int, dataset_size: int, 
                          available_memory_gb: float = None) -> Dict[str, Any]:
        """
        Get optimal DataLoader parameters
        
        Args:
            batch_size: Desired batch size
            dataset_size: Size of the dataset
            available_memory_gb: Available system memory in GB
            
        Returns:
            Dictionary with optimal DataLoader parameters
        """
        if available_memory_gb is None:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Estimate optimal number of workers
        cpu_count = os.cpu_count()
        
        # Conservative estimate: 1 worker per 2 GB of available memory
        max_workers_memory = max(1, int(available_memory_gb // 2))
        max_workers_cpu = max(1, cpu_count - 1)  # Leave one CPU core free
        
        num_workers = min(16, max_workers_memory, max_workers_cpu)
        
        # Adjust based on dataset size
        if dataset_size < 1000:
            num_workers = min(num_workers, 2)  # Small datasets don't need many workers
        
        # Other optimizations
        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0 and dataset_size > 5000
        prefetch_factor = 2 if num_workers > 0 else None
        
        params = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'prefetch_factor': prefetch_factor,
            'drop_last': False,
            'timeout': 300 if num_workers > 0 else 0  # 5 minute timeout
        }
        
        return params


class GradientCheckpointing:
    """
    Utilities for gradient checkpointing to save memory during training
    Author: Siddik
    """
    
    @staticmethod
    def checkpoint_sequential(functions, segments, *inputs):
        """
        Checkpoint sequential computation to save memory
        
        Args:
            functions: List of functions to apply sequentially
            segments: Number of segments to checkpoint
            *inputs: Input tensors
            
        Returns:
            Output of sequential computation
        """
        def run_function(start, end, functions):
            def forward(*inputs):
                for j in range(start, end + 1):
                    inputs = functions[j](*inputs)
                return inputs
            return forward
        
        if isinstance(functions, torch.nn.Sequential):
            functions = list(functions.children())
        
        segment_size = len(functions) // segments
        
        # Run first segment normally
        end = segment_size - 1
        if end >= 0:
            inputs = run_function(0, end, functions)(*inputs)
        
        # Checkpoint remaining segments
        for start in range(segment_size, len(functions), segment_size):
            end = min(start + segment_size - 1, len(functions) - 1)
            inputs = torch.utils.checkpoint.checkpoint(
                run_function(start, end, functions), *inputs
            )
        
        return inputs


def optimize_for_memory(config, model, dataset_size: int) -> Dict[str, Any]:
    """
    Get comprehensive memory optimization recommendations
    
    Args:
        config: Model configuration
        model: PyTorch model
        dataset_size: Size of the dataset
        
    Returns:
        Dictionary with optimization recommendations
    """
    logger = logging.getLogger(__name__)
    
    # Get system info
    memory_info = psutil.virtual_memory()
    available_memory_gb = memory_info.available / (1024**3)
    
    # Model memory estimation
    model_params = sum(p.numel() for p in model.parameters())
    model_memory_mb = model_params * 4 / (1024**2)  # Assume float32
    
    # Recommendations
    recommendations = {
        'model_memory_mb': model_memory_mb,
        'available_memory_gb': available_memory_gb,
    }
    
    # DataLoader optimizations
    dataloader_params = DataLoaderOptimizer.get_optimal_params(
        config.training.batch_size, dataset_size, available_memory_gb
    )
    recommendations['dataloader'] = dataloader_params
    
    # Batch size recommendations
    max_batch_size = int(available_memory_gb * 1024 / 4)  # Very conservative estimate
    if config.training.batch_size > max_batch_size:
        recommendations['suggested_batch_size'] = max_batch_size
        logger.warning(f"Batch size {config.training.batch_size} may be too large. "
                      f"Consider reducing to {max_batch_size}")
    
    # Memory efficiency recommendations
    if available_memory_gb < 8:
        recommendations['enable_gradient_checkpointing'] = True
        recommendations['use_mixed_precision'] = True
        logger.warning("Low memory detected. Consider enabling gradient checkpointing and mixed precision.")
    
    # GPU memory optimization
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            recommendations[f'gpu_{i}_memory_gb'] = gpu_memory_gb
            
            if gpu_memory_gb < 6:
                logger.warning(f"GPU {i} has limited memory ({gpu_memory_gb:.1f} GB). "
                              "Consider reducing batch size or enabling mixed precision.")
    
    return recommendations


@contextmanager
def memory_profiling(name: str = "operation"):
    """Context manager for detailed memory profiling"""
    monitor = MemoryMonitor()
    
    try:
        with monitor.monitor_block(name):
            yield monitor
    finally:
        logger = logging.getLogger(__name__)
        logger.info(monitor.get_memory_summary())


# Memory cleanup utilities
def cleanup_memory():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_memory_efficient_defaults():
    """Set memory-efficient defaults for PyTorch"""
    # Limit number of threads to prevent memory explosion
    torch.set_num_threads(min(4, os.cpu_count()))
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except AttributeError:
        pass  # Feature not available in this PyTorch version