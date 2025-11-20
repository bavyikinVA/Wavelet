import numpy as np
from typing import Dict, Any
import logging
from multiprocessing import Pool

logger = logging.getLogger(__name__)

def _process_row_wrapper(args):
    from .cpu_wavelet import morlet_wavelet_with_padding
    row_data, scales = args
    return morlet_wavelet_with_padding(row_data, scales)

class ComputeBackend:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.gpu_processor = None
        self.backend_info = {}
        self._initialize_backend()

    def _initialize_backend(self):
        self.backend_info = {
            "use_gpu": False,
            "device_name": "CPU",
            "gpu_memory": "N/A",
            "status": "active",
            "available": True
        }

        try:
            from .cpu_wavelet import morlet_wavelet_with_padding
            self._cpu_processor = morlet_wavelet_with_padding
            logger.info("CPU backend initialized")
        except Exception as e:
            logger.error(f"CPU backend initialization failed: {e}")
            self.backend_info["available"] = False
            return

        if self.use_gpu:
            try:
                from .gpu_processor import GPUWaveletProcessor
                self.gpu_processor = GPUWaveletProcessor(accuracy_mode="balanced")

                if self.gpu_processor.is_available():
                    gpu_info = self.gpu_processor.get_gpu_info()
                    self.backend_info.update({
                        "use_gpu": True,
                        "device_name": gpu_info.get("device_name", "CuPy GPU"),
                        "gpu_memory": f"{gpu_info.get('memory_free_mb', 0)}/{gpu_info.get('memory_total_mb', 0)} MB",
                        "backend": "CuPy"
                    })
                    logger.info(f"GPU backend: {self.backend_info['device_name']}")
                else:
                    logger.info("🔧 GPU not available, using CPU")
                    self.use_gpu = False

            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                self.use_gpu = False

    def morlet_wavelet_batch(self, data: np.ndarray, scales: np.ndarray) -> np.ndarray:
        if not self.backend_info["available"]:
            raise RuntimeError("No compute backend available")

        if self.use_gpu and self.gpu_processor and self.gpu_processor.is_available():
            try:
                logger.debug(f"GPU processing: {data.shape} signals")
                result = self.gpu_processor.morlet_wavelet_batch(data, scales)
                return result
            except Exception as e:
                logger.warning(f"GPU computation failed: {e}. Using CPU.")
                return self._compute_cpu_fallback(data, scales)
        else:
            return self._compute_cpu_fallback(data, scales)

    def _compute_cpu_fallback(self, data: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """CPU fallback implementation without pickle issues"""
        logger.info(f"CPU processing: {data.shape} signals with {len(scales)} scales")

        try:
            logger.debug("Starting multiprocessing pool...")
            with Pool() as pool:
                args = [(data[i], scales) for i in range(data.shape[0])]
                logger.debug(f"Processing {len(args)} items...")
                results = pool.map(_process_row_wrapper, args)

            logger.info("CPU multiprocessing completed successfully")
            return np.array(results)

        except Exception as e:
            logger.error(f"CPU multiprocessing failed: {e}")
            return self._compute_cpu_sequential(data, scales)

    def _compute_cpu_sequential(self, data: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Sequential CPU computation as last resort"""
        logger.warning("Using sequential CPU computation")
        results = []
        for i in range(data.shape[0]):
            result = self._cpu_processor(data[i], scales)
            results.append(result)
        return np.array(results)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend"""
        return self.backend_info.copy()

    def is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        return (self.gpu_processor is not None and
                self.gpu_processor.is_available())

    def toggle_backend(self) -> bool:
        """Toggle between CPU and GPU backend"""
        if self.use_gpu:
            # Switch to CPU
            self.use_gpu = False
            self.backend_info.update({
                "use_gpu": False,
                "device_name": "CPU",
                "gpu_memory": "N/A"
            })
            logger.info("Switched to CPU backend")
        else:
            # Try to switch to GPU
            if self.is_gpu_available():
                self.use_gpu = True
                gpu_info = self.gpu_processor.get_gpu_info()
                self.backend_info.update({
                    "use_gpu": True,
                    "device_name": gpu_info.get("device_name", "CuPy GPU"),
                    "gpu_memory": f"{gpu_info.get('memory_free_mb', 0)}/{gpu_info.get('memory_total_mb', 0)} MB"
                })
                logger.info(f"Switched to GPU backend: {self.backend_info['device_name']}")
            else:
                logger.warning("GPU not available, staying on CPU")

        return self.use_gpu

    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if self.gpu_processor:
            try:
                self.gpu_processor.clear_cache()
            except Exception as e:
                logger.error(f"GPU cache clear failed: {e}")