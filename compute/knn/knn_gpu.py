import logging
from typing import Dict, Optional

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


class KNN_GPU:
    def __init__(self):
        self.cp = cp
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Проверка доступности GPU"""
        try:
            # Простой тест вычислений на GPU
            test_array = self.cp.array([1.0, 2.0, 3.0])
            result = self.cp.sum(test_array)
            return True
        except Exception as e:
            logger.warning(f"GPU недоступен для KNN: {e}")
            return False

    def find_k_nearest_neighbors(self, points: np.ndarray, k: int) -> Optional[Dict]:
        """Поиск k-ближайших соседей на GPU"""
        if not self._available or len(points) < 2:
            return None

        try:
            # Автоматическое уменьшение k если нужно
            if len(points) <= k:
                k = len(points) - 1

            # Перенос данных на GPU
            points_gpu = self.cp.asarray(points, dtype=self.cp.float32)

            # Безопасное вычисление расстояний
            points_sq = self.cp.sum(points_gpu ** 2, axis=1)
            dot_product = points_gpu @ points_gpu.T

            # Стабильное вычисление расстояний
            dist_matrix = points_sq[:, None] + points_sq[None, :] - 2 * dot_product

            # Защита от численных ошибок
            dist_matrix = self.cp.maximum(dist_matrix, 0.0)

            # Поиск k+1 ближайших соседей (включая саму точку)
            k_actual = min(k + 1, len(points))
            indices = self.cp.argsort(dist_matrix, axis=1)[:, :k_actual]
            distances = self.cp.sqrt(self.cp.take_along_axis(dist_matrix, indices, axis=1))

            # Конвертация в CPU и форматирование результата
            indices_cpu = self.cp.asnumpy(indices)
            distances_cpu = self.cp.asnumpy(distances)

            neighbors_dict = {}
            for i in range(len(points)):
                # Исключаем саму точку (первый элемент)
                start_idx = 1 if k_actual > 1 else 0
                neighbors_dict[i] = {
                    'indices': indices_cpu[i][start_idx:].tolist(),
                    'distances': distances_cpu[i][start_idx:].tolist()
                }

            return neighbors_dict

        except Exception as e:
            logger.error(f"Ошибка GPU KNN: {e}")
            return None

    def is_available(self) -> bool:
        return self._available

    def clear_cache(self):
        """Очистка памяти GPU"""
        try:
            self.cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            logger.debug(f"Ошибка очистки кэша GPU: {e}")