# compute/cupy_wavelet.py - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ

import numpy as np
import logging
import cupy as cp

logger = logging.getLogger(__name__)


class CupyWaveletGPU:
    def __init__(self):
        self.cp = cp
        self._available = True
        self._compile_kernels()

    def _compile_kernels(self):
        """Compile optimized CUDA kernels"""
        self.single_scale_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void morlet_single_scale(const float* data, float* result, int data_len, 
                                float scale, int pad_width) {
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (j >= data_len) return;

            float w0 = 0.0f;
            float inv_scale = 1.0f / scale;
            float sqrt_scale = sqrtf(scale);

            for (int k = -pad_width; k < data_len + pad_width; k++) {
                int actual_k = k;
                // Symmetric reflection
                if (k < 0) {
                    actual_k = -k - 1;
                } else if (k >= data_len) {
                    actual_k = 2 * data_len - k - 1;
                }

                float t = (k - j) * inv_scale;
                float exp_val = expf(-(t * t) * 0.5f);
                float cos_val = cosf(2.0f * 3.14159265f * t);
                w0 += data[actual_k] * 0.75f * exp_val * cos_val;
            }

            result[j] = w0 / sqrt_scale;
        }
        ''', 'morlet_single_scale')

    def compute_batch_signals(self, data_batch: np.ndarray, scales: np.ndarray) -> np.ndarray:
        data_batch = data_batch.astype(np.float32)
        scales = scales.astype(np.float32)

        num_signals, signal_length = data_batch.shape
        num_scales = len(scales)

        logger.info(f"GPU processing: {num_signals} signals × {signal_length} points × {num_scales} scales")

        # Если данные слишком большие, обрабатываем частями
        if num_signals * signal_length * num_scales > 1000000:
            return self._compute_batch_chunked(data_batch, scales)

        results = []

        for signal_idx in range(num_signals):
            signal_data = data_batch[signal_idx]
            signal_result = self.cp.zeros((num_scales, signal_length), dtype=self.cp.float32)

            signal_gpu = self.cp.asarray(signal_data)

            for scale_idx, scale in enumerate(scales):
                pad_width = int(7 * scale) // 2 + 1
                result_scale = self.cp.zeros(signal_length, dtype=self.cp.float32)

                # Запускаем kernel
                block_size = 256
                grid_size = (signal_length + block_size - 1) // block_size

                self.single_scale_kernel(
                    (grid_size,), (block_size,),
                    (signal_gpu, result_scale, signal_length, scale, pad_width)
                )

                signal_result[scale_idx] = result_scale

            results.append(self.cp.asnumpy(signal_result))

        return np.array(results)

    def _compute_batch_chunked(self, data_batch: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Обработка больших батчей по частям"""
        num_signals, signal_length = data_batch.shape
        num_scales = len(scales)

        chunk_size = max(1, min(10, num_signals // 4))  # Адаптивный размер чанка
        results = []

        for chunk_start in range(0, num_signals, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_signals)
            chunk_data = data_batch[chunk_start:chunk_end]

            logger.info(f"Processing chunk {chunk_start}-{chunk_end} of {num_signals}")

            chunk_results = []
            for signal_idx in range(len(chunk_data)):
                signal_data = chunk_data[signal_idx]
                signal_result = self.cp.zeros((num_scales, signal_length), dtype=self.cp.float32)

                signal_gpu = self.cp.asarray(signal_data)

                for scale_idx, scale in enumerate(scales):
                    pad_width = int(7 * scale) // 2 + 1
                    result_scale = self.cp.zeros(signal_length, dtype=self.cp.float32)

                    block_size = 256
                    grid_size = (signal_length + block_size - 1) // block_size

                    self.single_scale_kernel(
                        (grid_size,), (block_size,),
                        (signal_gpu, result_scale, signal_length, scale, pad_width)
                    )

                    signal_result[scale_idx] = result_scale

                chunk_results.append(self.cp.asnumpy(signal_result))

            results.extend(chunk_results)

            # Очищаем память после каждого чанка
            self.cp.get_default_memory_pool().free_all_blocks()

        return np.array(results)

    def is_available(self):
        return self._available

    def get_gpu_info(self):
        """Get GPU information"""
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
        try:
            self.cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            logger.debug(f"Cache clearing failed: {e}")