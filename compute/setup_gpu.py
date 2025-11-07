"""
Setup script for CUDA environment
"""

import os
import sys


def setup_cuda_environment():
    """Setup CUDA environment variables for PyCUDA"""

    # Possible CUDA paths
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        os.environ.get('CUDA_PATH'),
        os.environ.get('CUDA_HOME'),
    ]

    found_path = None
    for path in cuda_paths:
        if path and os.path.exists(path):
            if os.path.exists(os.path.join(path, 'bin', 'nvcc.exe')):
                found_path = path
                print(f"✅ Found CUDA at: {path}")
                break

    if found_path:
        # Set environment variables
        os.environ['CUDA_PATH'] = found_path
        cuda_bin = os.path.join(found_path, 'bin')
        cuda_lib = os.path.join(found_path, 'lib', 'x64')

        # Add to PATH
        current_path = os.environ.get('PATH', '')
        if cuda_bin not in current_path:
            os.environ['PATH'] = cuda_bin + os.pathsep + current_path
        if cuda_lib not in current_path:
            os.environ['PATH'] = cuda_lib + os.pathsep + current_path

        print(f"🔧 Configured CUDA_PATH: {found_path}")
        return True
    else:
        print("❌ CUDA path not found")
        return False


if __name__ == "__main__":
    setup_cuda_environment()