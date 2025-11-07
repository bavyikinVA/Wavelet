"""
OPTIMIZED CuPy-based GPU wavelet implementation
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class CupyWaveletGPU:
    """
    OPTIMIZED GPU wavelet processing using CuPy with matrix operations
    """

    def __init__(self):
        self.cp = None
        self._available = False
        self._initialize_cupy()

    def _initialize_cupy(self):
        """Initialize CuPy with error handling"""
        try:
            import cupy as cp
            self.cp = cp

            # Test basic functionality
            x = cp.arange(10, dtype=cp.float64)
            y = cp.sin(x)
            _ = cp.asnumpy(y)

            self._available = True
            logger.info("✅ CuPy GPU backend initialized successfully")

        except ImportError:
            logger.warning("❌ CuPy not available")
            self._available = False
        except Exception as e:
            logger.warning(f"❌ CuPy initialization failed: {e}")
            self._available = False

    def compute_batch_signals_fast(self, data_batch: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        FAST vectorized implementation using matrix operations
        """
        if not self.is_available():
            raise RuntimeError("CuPy not available")

        # Use float32 for better performance (adequate for most applications)
        data_batch = data_batch.astype(np.float32)
        scales = scales.astype(np.float32)

        num_signals, signal_length = data_batch.shape
        num_scales = len(scales)

        logger.info(f"🚀 FAST GPU processing: {num_signals} signals × {signal_length} points × {num_scales} scales")

        # Transfer to GPU
        data_gpu = self.cp.asarray(data_batch)
        scales_gpu = self.cp.asarray(scales)

        results = []

        for signal_idx in range(num_signals):
            signal_data = data_gpu[signal_idx]
            signal_result = self.cp.zeros((num_scales, signal_length), dtype=self.cp.float32)

            # Precompute time array
            t = self.cp.arange(signal_length, dtype=self.cp.float32)

            # Create position matrix [signal_length, signal_length]
            t_matrix = t[:, None] - t[None, :]

            for scale_idx, scale in enumerate(scales_gpu):
                # Vectorized Morlet wavelet computation
                t_scaled = t_matrix / scale

                # Morlet wavelet kernel
                wavelet_matrix = 0.75 * self.cp.exp(-0.5 * t_scaled**2) * self.cp.cos(2 * self.cp.pi * t_scaled)

                # Apply transform (matrix multiplication)
                signal_result[scale_idx] = self.cp.dot(signal_data, wavelet_matrix) / self.cp.sqrt(scale)

            results.append(self.cp.asnumpy(signal_result))

        return np.array(results)

    def compute_batch_signals_optimized(self, data_batch: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        OPTIMIZED version with chunk processing for large datasets
        """
        if not self.is_available():
            raise RuntimeError("CuPy not available")

        data_batch = data_batch.astype(np.float32)
        scales = scales.astype(np.float32)

        num_signals, signal_length = data_batch.shape
        num_scales = len(scales)

        logger.info(f"⚡ OPTIMIZED GPU processing: {num_signals} signals")

        # Process in chunks to manage memory
        chunk_size = min(5, num_signals)  # Smaller chunks for stability
        all_results = []

        for chunk_start in range(0, num_signals, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_signals)
            chunk_data = data_batch[chunk_start:chunk_end]

            # Process chunk using fast method
            chunk_results = self._process_chunk_fast(chunk_data, scales, signal_length)
            all_results.extend(chunk_results)

        return np.array(all_results)

    def _process_chunk_fast(self, chunk_data: np.ndarray, scales: np.ndarray, signal_length: int) -> list:
        """Process a chunk of data using fast matrix operations"""
        import cupy as cp

        chunk_gpu = cp.asarray(chunk_data)
        scales_gpu = cp.asarray(scales)

        chunk_results = []

        for i in range(len(chunk_data)):
            signal_data = chunk_gpu[i]
            signal_result = cp.zeros((len(scales), signal_length), dtype=cp.float32)

            # Precompute time matrix once per signal
            t = cp.arange(signal_length, dtype=cp.float32)
            t_matrix = t[:, None] - t[None, :]

            for scale_idx, scale in enumerate(scales_gpu):
                t_scaled = t_matrix / scale
                wavelet_matrix = 0.75 * cp.exp(-0.5 * t_scaled**2) * cp.cos(2 * cp.pi * t_scaled)
                signal_result[scale_idx] = cp.dot(signal_data, wavelet_matrix) / cp.sqrt(scale)

            chunk_results.append(cp.asnumpy(signal_result))

        return chunk_results

    def compute_batch_signals(self, data_batch: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Main method - uses optimized version by default
        """
        return self.compute_batch_signals_optimized(data_batch, scales)

    def is_available(self):
        return self._available

    def get_gpu_info(self):
        """Get GPU information"""
        if not self.is_available():
            return {"available": False}

        try:
            mem_info = self.cp.cuda.Device().mem_info
            return {
                "available": True,
                "memory_free_mb": mem_info[0] // (1024 ** 2),
                "memory_total_mb": mem_info[1] // (1024 ** 2),
                "device_name": "CuPy GPU",
                "backend": "CuPy"
            }
        except Exception as e:
            return {
                "available": True,
                "device_name": "CuPy GPU",
                "error": str(e)
            }

    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.is_available():
            try:
                self.cp.get_default_memory_pool().free_all_blocks()
                import gc
                gc.collect()
            except Exception as e:
                logger.debug(f"Cache clearing failed: {e}")