# test_cupy_cuda12.py
import os

# Настройка перед импортом
print("Настройка окружения...")

# Указываем использовать драйвер (не toolkit)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUPY_CACHE_DIR'] = os.path.join(os.path.expanduser('~'), '.cupy', 'cache_cuda12')

try:
    import cupy as cp

    print("CuPy успешно импортирован.")
    print(f"   Версия CuPy: {cp.__version__}")

    # Информация о GPU
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f"Доступно GPU: {device_count}")

    for i in range(device_count):
        props = cp.cuda.runtime.getDeviceProperties(i)
        print(f"   GPU {i}: {props.get('name', 'Unknown')}")

    # Простой тест
    print("Тестируем вычисления...")
    x = cp.arange(10, dtype=cp.float32)
    y = cp.sin(x)
    result = y.get()
    print(f"GPU вычисления работают: {result}")

    # Тест с матрицами
    print("Тестируем матричные операции...")
    a = cp.random.random((100, 100), dtype=cp.float32)
    b = cp.random.random((100, 100), dtype=cp.float32)
    c = cp.dot(a, b)
    print(f"Матричные операции работают: результат shape {c.shape}")

    print("CuPy полностью работает с GPU.")

except ImportError as e:
    print(f"Ошибка импорта CuPy: {e}")
    print("Попробуйте: pip install cupy-cuda12x")
except Exception as e:
    print(f"Ошибка работы CuPy: {e}")
