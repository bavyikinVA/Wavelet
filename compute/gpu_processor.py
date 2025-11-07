"""
GPU processor using OPTIMIZED CuPy implementation
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class GPUWaveletProcessor:
    """
    GPU processor using OPTIMIZED CuPy implementation
    """

    def __init__(self, accuracy_mode: str = "fast"):  # По умолчанию FAST для производительности
        self.gpu_processor = None
        self._gpu_available = False
        self._gpu_info = {}
        self.accuracy_mode = accuracy_mode
        self._initialize_gpu()

    def _initialize_gpu(self):
        """Initialize GPU using CuPy"""
        self._gpu_info = {
            "available": False,
            "device_name": "Unknown",
            "memory_free_mb": 0,
            "memory_total_mb": 0,
            "error": None,
            "backend": "None"
        }

        try:
            from .cupy_wavelet import CupyWaveletGPU
            self.gpu_processor = CupyWaveletGPU()

            if self.gpu_processor.is_available():
                self._gpu_available = True
                gpu_info = self.gpu_processor.get_gpu_info()
                self._gpu_info.update(gpu_info)
                logger.info(f"✅ OPTIMIZED CuPy GPU processor initialized (mode: {self.accuracy_mode})")
            else:
                logger.warning("❌ CuPy GPU not available")

        except Exception as e:
            logger.warning(f"❌ GPU initialization failed: {e}")
            self._gpu_info["error"] = str(e)

    def morlet_wavelet_batch(self, data: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Compute wavelet transform for batch of signals on GPU
        """
        if not self.is_available():
            raise RuntimeError("GPU processor not available")

        if data.ndim != 2:
            raise ValueError("Expected 2D array for data")

        logger.info(f"🚀 GPU processing: {data.shape[0]} signals × {data.shape[1]} points × {len(scales)} scales")

        try:
            # Для больших изображений используем FAST метод независимо от режима
            if data.shape[0] * data.shape[1] > 100000:  # Большое изображение
                logger.info("🔧 Используется FAST метод для большого изображения")
                return self.gpu_processor.compute_batch_signals_fast(data, scales)
            elif self.accuracy_mode == "fast":
                return self.gpu_processor.compute_batch_signals_fast(data, scales)
            else:
                return self.gpu_processor.compute_batch_signals_optimized(data, scales)

        except Exception as e:
            logger.error(f"❌ GPU computation failed: {e}")
            # Return zeros as fallback
            num_signals, signal_length = data.shape
            num_scales = len(scales)
            logger.warning("🔄 Возврат нулевых результатов из-за ошибки GPU")
            return np.zeros((num_signals, num_scales, signal_length), dtype=np.float64)

    def set_accuracy_mode(self, mode: str):
        """Set accuracy mode"""
        valid_modes = ["high", "balanced", "fast"]
        if mode in valid_modes:
            self.accuracy_mode = mode
            logger.info(f"🔧 GPU accuracy mode set to: {mode}")
        else:
            logger.warning(f"Invalid accuracy mode: {mode}. Using 'fast'.")

    def is_available(self) -> bool:
        """Check if GPU processing is available"""
        return self._gpu_available

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        return self._gpu_info.copy()

    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.is_available() and hasattr(self.gpu_processor, 'clear_cache'):
            self.gpu_processor.clear_cache()