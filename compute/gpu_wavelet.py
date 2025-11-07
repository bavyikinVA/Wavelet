"""
Simplified GPU implementation with better error handling
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import GPU libraries with comprehensive error handling
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    PYCUDA_AVAILABLE = True
    logger.info("✅ PyCUDA successfully imported")
except ImportError as e:
    logger.warning(f"❌ PyCUDA import failed: {e}")
    PYCUDA_AVAILABLE = False

class MorletWaveletGPU:
    """
    Simplified GPU implementation with robust fallbacks
    """

    def __init__(self, block_size: int = 256):
        self.block_size = block_size
        self.mod = None
        self.kernel = None
        self._compilation_attempted = False

        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        # Don't compile immediately, wait for first use
        self._compile_on_demand = True

    def _get_safe_compute_capability(self):
        """Get compute capability with fallbacks"""
        try:
            device = cuda.Context.get_device()
            major = device.compute_capability()[0]
            minor = device.compute_capability()[1]

            # Map to supported architectures
            if major >= 8:
                return "sm_80"  # Common for Ampere and later
            elif major >= 7:
                return "sm_70"  # Volta and Turing
            else:
                return "sm_60"  # Pascal and later

        except Exception as e:
            logger.warning(f"Could not detect compute capability: {e}")
            return "sm_60"  # Safe fallback

    def _compile_kernels_safe(self):
        """Safe kernel compilation with multiple fallbacks"""
        if self._compilation_attempted:
            return self.mod is not None

        self._compilation_attempted = True

        # Simplified kernel code without complex C++ features
        kernel_code = """
extern "C" {

__global__ void morlet_wavelet_kernel(
    const float* data, 
    float* result, 
    const float* scales,
    int data_len, 
    int num_scales,
    int pad_width) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int scale_idx = blockIdx.y;
    
    if (j < data_len && scale_idx < num_scales) {
        float scale = scales[scale_idx];
        float w0 = 0.0f;
        float inv_scale = 1.0f / scale;
        float sqrt_scale = sqrtf(scale);
        
        for (int k = -pad_width; k < data_len + pad_width; k++) {
            int actual_k = k;
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
        
        result[scale_idx * data_len + j] = w0 / sqrt_scale;
    }
}

}
"""

        compilation_attempts = [
            # Try with detected architecture
            ['-arch=' + self._get_safe_compute_capability(), '-use_fast_math'],
            # Try common architectures
            ['-arch=sm_80', '-use_fast_math'],
            ['-arch=sm_75', '-use_fast_math'],
            ['-arch=sm_70', '-use_fast_math'],
            ['-arch=sm_60', '-use_fast_math'],
            # Try without specific architecture
            ['-use_fast_math'],
            # Minimal attempt
            []
        ]

        for attempt, options in enumerate(compilation_attempts):
            try:
                logger.info(f"🔄 Compilation attempt {attempt + 1} with options: {options}")
                self.mod = SourceModule(kernel_code, options=options, no_extern_c=False)
                self.kernel = self.mod.get_function("morlet_wavelet_kernel")
                logger.info(f"✅ CUDA kernel compiled successfully with options: {options}")
                return True

            except Exception as e:
                logger.warning(f"❌ Compilation attempt {attempt + 1} failed: {e}")
                continue

        logger.error("❌ All compilation attempts failed")
        return False

    def compute_batch_signals(self, data_batch: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Batch processing of multiple signals on GPU
        """
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        # Compile on first use
        if self._compile_on_demand:
            if not self._compile_kernels_safe():
                raise RuntimeError("Failed to compile CUDA kernels")
            self._compile_on_demand = False

        if self.kernel is None:
            raise RuntimeError("Kernel not available")

        data_batch = data_batch.astype(np.float32)
        scales = scales.astype(np.float32)

        num_signals, signal_length = data_batch.shape
        num_scales = len(scales)

        # Calculate padding
        max_scale = np.max(scales)
        pad_width = int(7 * max_scale) // 2 + 1

        logger.info(f"🎮 GPU processing: {num_signals} signals")

        # Process each signal
        all_results = []

        for signal_idx in range(num_signals):
            signal_data = data_batch[signal_idx]

            # Allocate GPU memory
            data_gpu = gpuarray.to_gpu(signal_data)
            scales_gpu = gpuarray.to_gpu(scales)
            result_gpu = gpuarray.zeros((num_scales * signal_length), dtype=np.float32)

            # Configure kernel execution
            block_x = min(self.block_size, signal_length)
            grid_x = (signal_length + block_x - 1) // block_x
            grid_y = num_scales

            try:
                # Execute kernel
                self.kernel(
                    data_gpu, result_gpu, scales_gpu,
                    np.int32(signal_length), np.int32(num_scales), np.int32(pad_width),
                    block=(block_x, 1, 1),
                    grid=(grid_x, grid_y, 1)
                )

                # Get result
                signal_result = result_gpu.get().reshape(num_scales, signal_length)
                all_results.append(signal_result)

            except Exception as e:
                logger.error(f"❌ Kernel execution failed: {e}")
                # Return zeros as fallback
                all_results.append(np.zeros((num_scales, signal_length), dtype=np.float32))

        return np.array(all_results)

    def is_available(self):
        """Check if GPU processing is available"""
        if not PYCUDA_AVAILABLE:
            return False
        if not self._compilation_attempted:
            return self._compile_kernels_safe()
        return self.kernel is not None

    def clear_cache(self):
        """Clear resources"""
        try:
            if hasattr(self, 'mod'):
                del self.mod
            if hasattr(self, 'kernel'):
                del self.kernel
            import gc
            gc.collect()
        except:
            pass