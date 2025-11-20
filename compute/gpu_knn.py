import numpy as np
import cupy as cp
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class KNN:
    def __init__(self):
        self.cp = cp

    def find_k_nearest_neighbors_gpu(self, points: np.ndarray, k: int) -> Dict:
        if len(points) < 2:
            return {}

        if len(points) <= k:
            k = len(points) - 1

        # Transfer to GPU
        points_gpu = self.cp.asarray(points, dtype=self.cp.float32)

        # Compute pairwise distances using matrix operations
        points_sq = self.cp.sum(points_gpu ** 2, axis=1)
        dist_matrix = points_sq[:, None] + points_sq[None, :] - 2 * points_gpu @ points_gpu.T

        # Find k+1 nearest neighbors (including self)
        indices = self.cp.argsort(dist_matrix, axis=1)[:, :k + 1]
        distances = self.cp.take_along_axis(dist_matrix, indices, axis=1)

        # Convert to dictionary format
        indices_cpu = self.cp.asnumpy(indices)
        distances_cpu = self.cp.asnumpy(distances)

        neighbors_dict = {}
        for i in range(len(points)):
            # Exclude self (first element)
            neighbors_dict[i] = {
                'indices': indices_cpu[i][1:].tolist(),
                'distances': distances_cpu[i][1:].tolist()
            }

        return neighbors_dict

    def batch_process_extremes_gpu(self, extremes_list: List, k: int) -> List:
        results = []

        for extreme_dict in extremes_list:
            points = np.array(extreme_dict['points'])
            if len(points) >= 2:
                neighbors_dict = self.find_k_nearest_neighbors_gpu(points, k)
                results.append({
                    'extreme_dict': extreme_dict,
                    'neighbors_dict': neighbors_dict
                })
            else:
                results.append({
                    'extreme_dict': extreme_dict,
                    'neighbors_dict': {}
                })
        return results