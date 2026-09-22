import numpy as np
from typing import Dict, Any
import logging

from compute.backend_policy import (
    GPUUnavailableError,
    classify_gpu_exception,
)
from compute.numerics import COMPUTE_DTYPE

logger = logging.getLogger(__name__)


class GPUWaveletProcessor:
    def __init__(self):
        self.gpu_processor = None
        self._gpu_available = False
        self._gpu_info = {}
        self._initialize_gpu()

    def _initialize_gpu(self):
        """Initialize GPU using CuPy and verify a real CUDA operation."""
        self._gpu_info = {
            "available": False,
            "device_name": "Unknown",
            "memory_free_mb": 0,
            "memory_total_mb": 0,
            "error": None,
            "backend": "None",
        }

        try:
            from compute.wavelets.cupy_wavelet import CupyWaveletGPU
            self.gpu_processor = CupyWaveletGPU()
            if self.gpu_processor.is_available():
                self._gpu_available = True
                self._gpu_info.update(self.gpu_processor.get_gpu_info())
                logger.info("CuPy GPU processor initialized")
            else:
                logger.warning("CuPy GPU not available")
        except Exception as exc:
            mapped = classify_gpu_exception(exc)
            logger.warning("GPU initialization failed: %s", exc)
            self._gpu_info["error"] = str(exc)
            if mapped is not None:
                self._gpu_info["error_type"] = mapped.__class__.__name__

    def morlet_wavelet_batch(self, data: np.ndarray, scales: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise GPUUnavailableError("GPU processor not available")
        if data.ndim != 2:
            raise ValueError("Expected 2D array for data")

        data = np.asarray(data, dtype=COMPUTE_DTYPE)
        scales = np.asarray(scales, dtype=COMPUTE_DTYPE)
        logger.info(
            "GPU processing: %s signals × %s points × %s scales",
            data.shape[0], data.shape[1], len(scales),
        )

        try:
            return np.asarray(
                self.gpu_processor.compute_batch_signals(data, scales),
                dtype=COMPUTE_DTYPE,
            )
        except Exception as exc:
            mapped = classify_gpu_exception(exc)
            if mapped is None:
                logger.exception("GPU wavelet algorithm failed")
                raise
            logger.error("GPU wavelet infrastructure failure: %s", mapped)
            raise mapped from exc

    def is_available(self) -> bool:
        return self._gpu_available

    def get_gpu_info(self) -> Dict[str, Any]:
        return self._gpu_info.copy()

    def clear_cache(self):
        if self.is_available() and hasattr(self.gpu_processor, "clear_cache"):
            self.gpu_processor.clear_cache()
