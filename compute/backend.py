import logging
from multiprocessing import Pool
from typing import Any, Dict

import numpy as np

from compute.backend_policy import (
    ExecutionProtocol,
    GPU_FALLBACK_ERRORS,
    GPUUnavailableError,
)
from compute.numerics import COMPUTE_DTYPE, COMPUTE_DTYPE_NAME

logger = logging.getLogger(__name__)


def _process_row_wrapper(args):
    from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding
    row_data, scales = args
    return morlet_wavelet_with_padding(row_data, scales)


class ComputeBackend:
    def __init__(self, use_gpu: bool = True, *, lazy: bool = False,
                 strict_backend: bool = False):
        self.use_gpu = use_gpu
        self.requested_backend = "gpu" if use_gpu else "cpu"
        self.strict_backend = bool(strict_backend)
        self.gpu_processor = None
        self._initialized = False
        self._gpu_checked = False
        self.backend_info = {
            "use_gpu": use_gpu,
            "requested_backend": self.requested_backend,
            "device_name": "Авто (GPU / CPU)" if use_gpu else "CPU",
            "gpu_available": False,
            "gpu_device_name": "GPU",
            "gpu_memory": "N/A",
            "status": "pending",
            "available": True,
            "dtype": COMPUTE_DTYPE_NAME,
            "strict_backend": self.strict_backend,
        }
        if not lazy:
            self.ensure_initialized()

    def ensure_initialized(self):
        if not self._initialized:
            self._initialize_backend()
            self._initialized = True

    def _initialize_backend(self):
        self.backend_info = {
            "use_gpu": False,
            "requested_backend": self.requested_backend,
            "device_name": "CPU",
            "gpu_available": False,
            "gpu_device_name": "GPU",
            "gpu_memory": "N/A",
            "status": "active",
            "available": True,
            "dtype": COMPUTE_DTYPE_NAME,
            "strict_backend": self.strict_backend,
        }

        try:
            from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding
            self._cpu_processor = morlet_wavelet_with_padding
            logger.info("CPU backend initialized (%s)", COMPUTE_DTYPE_NAME)
        except Exception as exc:
            logger.error("CPU backend initialization failed: %s", exc)
            self.backend_info["available"] = False
            return

        if self.use_gpu:
            self._gpu_checked = True
            try:
                from compute.wavelets.gpu_processor import GPUWaveletProcessor
                self.gpu_processor = GPUWaveletProcessor()
                if self.gpu_processor.is_available():
                    gpu_info = self.gpu_processor.get_gpu_info()
                    self.backend_info.update({
                        "use_gpu": True,
                        "device_name": gpu_info.get("device_name", "CuPy GPU"),
                        "gpu_available": True,
                        "gpu_device_name": gpu_info.get("device_name", "CuPy GPU"),
                        "gpu_memory": f"{gpu_info.get('memory_free_mb', 0)}/{gpu_info.get('memory_total_mb', 0)} MB",
                        "backend": "CuPy",
                    })
                    logger.info("GPU backend: %s", self.backend_info["device_name"])
                else:
                    if self.strict_backend:
                        raise GPUUnavailableError("Requested GPU backend is unavailable")
                    logger.info("GPU not available, using CPU")
                    self.use_gpu = False
            except GPU_FALLBACK_ERRORS:
                raise
            except Exception as exc:
                if self.strict_backend:
                    raise GPUUnavailableError(str(exc)) from exc
                logger.warning("GPU initialization failed: %s", exc)
                self.use_gpu = False

    def morlet_wavelet_batch(self, data: np.ndarray, scales: np.ndarray,
                             *, protocol: ExecutionProtocol | None = None,
                             stage: str = "wavelet") -> np.ndarray:
        self.ensure_initialized()
        if not self.backend_info["available"]:
            raise RuntimeError("No compute backend available")

        data = np.asarray(data, dtype=COMPUTE_DTYPE)
        scales = np.asarray(scales, dtype=COMPUTE_DTYPE)

        if self.use_gpu and self.gpu_processor and self.gpu_processor.is_available():
            try:
                result = self.gpu_processor.morlet_wavelet_batch(data, scales)
                if protocol is not None:
                    protocol.record_stage(stage, actual_backend="gpu", dtype=COMPUTE_DTYPE_NAME)
                return np.asarray(result, dtype=COMPUTE_DTYPE)
            except GPU_FALLBACK_ERRORS as exc:
                if self.strict_backend or (protocol is not None and protocol.strict_backend):
                    raise
                logger.warning("GPU computation failed: %s. Using CPU.", exc)
                if protocol is not None:
                    protocol.record_fallback(
                        stage,
                        reason=exc.__class__.__name__,
                        message=str(exc),
                    )
                result = self._compute_cpu_fallback(data, scales)
                if protocol is not None:
                    protocol.record_stage(
                        stage,
                        actual_backend="cpu",
                        dtype=COMPUTE_DTYPE_NAME,
                        fallback=True,
                        details={"fallback_reason": exc.__class__.__name__},
                    )
                return result

        result = self._compute_cpu_fallback(data, scales)
        if protocol is not None:
            protocol.record_stage(stage, actual_backend="cpu", dtype=COMPUTE_DTYPE_NAME)
        return result

    def _compute_cpu_fallback(self, data: np.ndarray, scales: np.ndarray) -> np.ndarray:
        logger.info("CPU processing: %s signals with %s scales", data.shape, len(scales))
        data = np.asarray(data, dtype=COMPUTE_DTYPE)
        scales = np.asarray(scales, dtype=COMPUTE_DTYPE)
        try:
            with Pool() as pool:
                args = [(data[i], scales) for i in range(data.shape[0])]
                results = pool.map(_process_row_wrapper, args)
            return np.asarray(results, dtype=COMPUTE_DTYPE)
        except Exception as exc:
            logger.error("CPU multiprocessing failed: %s", exc)
            return self._compute_cpu_sequential(data, scales)

    def _compute_cpu_sequential(self, data: np.ndarray, scales: np.ndarray) -> np.ndarray:
        logger.warning("Using sequential CPU computation")
        results = [self._cpu_processor(data[i], scales) for i in range(data.shape[0])]
        return np.asarray(results, dtype=COMPUTE_DTYPE)

    def get_backend_info(self) -> Dict[str, Any]:
        if self.is_gpu_available():
            gpu_info = self.gpu_processor.get_gpu_info()
            self.backend_info.update({
                "gpu_available": True,
                "gpu_device_name": gpu_info.get("device_name", "CuPy GPU"),
                "gpu_memory": f"{gpu_info.get('memory_free_mb', 0)}/{gpu_info.get('memory_total_mb', 0)} MB",
            })
        self.backend_info.update({
            "requested_backend": self.requested_backend,
            "dtype": COMPUTE_DTYPE_NAME,
            "strict_backend": self.strict_backend,
        })
        return dict(self.backend_info, gpu_checked=self._gpu_checked)

    def is_gpu_available(self) -> bool:
        return self.gpu_processor is not None and self.gpu_processor.is_available()

    def set_use_gpu(self, enabled: bool) -> bool:
        self.requested_backend = "gpu" if enabled else "cpu"
        if not self._initialized or (enabled and not self._gpu_checked):
            self._initialized = False
            self.use_gpu = bool(enabled)
            self.backend_info.update(
                use_gpu=self.use_gpu,
                requested_backend=self.requested_backend,
                device_name="Авто (GPU / CPU)" if enabled else "CPU",
                status="pending",
            )
            return self.use_gpu
        if not enabled:
            self.use_gpu = False
            self.backend_info.update({"use_gpu": False, "device_name": "CPU"})
            logger.info("Switched to CPU backend")
        elif self.is_gpu_available():
            self.use_gpu = True
            gpu_info = self.gpu_processor.get_gpu_info()
            self.backend_info.update({
                "use_gpu": True,
                "device_name": gpu_info.get("device_name", "CuPy GPU"),
                "gpu_available": True,
                "gpu_device_name": gpu_info.get("device_name", "CuPy GPU"),
                "gpu_memory": f"{gpu_info.get('memory_free_mb', 0)}/{gpu_info.get('memory_total_mb', 0)} MB",
            })
            logger.info("Switched to GPU backend: %s", self.backend_info["device_name"])
        else:
            if self.strict_backend:
                raise GPUUnavailableError("Requested GPU backend is unavailable")
            self.use_gpu = False
            self.backend_info.update({"use_gpu": False, "device_name": "CPU", "gpu_available": False})
            logger.warning("GPU not available, staying on CPU")
        return self.use_gpu

    def set_strict_backend(self, enabled: bool) -> None:
        self.strict_backend = bool(enabled)
        self.backend_info["strict_backend"] = self.strict_backend

    def toggle_backend(self) -> bool:
        return self.set_use_gpu(not self.use_gpu)

    def clear_gpu_cache(self):
        if self.gpu_processor:
            try:
                self.gpu_processor.clear_cache()
            except Exception as exc:
                logger.error("GPU cache clear failed: %s", exc)
