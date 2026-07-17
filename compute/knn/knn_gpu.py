import logging
import time
import cupy as cp
import numpy as np

logger = logging.getLogger("WaveletApp")


class KNN_GPU:
    def __init__(
            self,
            query_batch_size: int = 1024,
            ref_batch_size: int = 2048,
            use_sqrt: bool = True,
            cleanup_each_ref_batch: bool = False,
            cleanup_each_query_batch: bool = False,
            log_timing: bool = True):
        self.cp = cp
        self.query_batch_size = query_batch_size
        self.ref_batch_size = ref_batch_size
        self.use_sqrt = use_sqrt
        self.cleanup_each_ref_batch = cleanup_each_ref_batch
        self.cleanup_each_query_batch = cleanup_each_query_batch
        self.log_timing = log_timing
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            test = self.cp.array([1.0, 2.0, 3.0])
            self.cp.sum(test)
            return True
        except Exception as e:
            logger.warning(f"GPU недоступен: {e}")
            return False

    def is_available(self) -> bool:
        return self._available

    def clear_cache(self):
        try:
            self.cp.get_default_memory_pool().free_all_blocks()
            self.cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            logger.error(f"{e}")

    def _sync(self):
        """Явная синхронизация GPU для корректного замера времени."""
        try:
            self.cp.cuda.Stream.null.synchronize()
        except Exception as e:
            logger.error(f"{e}")
            pass

    def _get_gpu_mem_info(self):
        """Возвращает информацию о памяти GPU."""
        try:
            free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
            pool = self.cp.get_default_memory_pool()
            return {
                "free_mb": free_bytes / (1024 ** 2),
                "total_mb": total_bytes / (1024 ** 2),
                "used_pool_mb": pool.used_bytes() / (1024 ** 2),
                "reserved_pool_mb": pool.total_bytes() / (1024 ** 2),
            }
        except Exception as e:
            print(e)
            return None

    def find_k_nearest_neighbors(self, points: np.ndarray, k: int):

        if not self._available or len(points) < 2:
            return None

        try:
            t_total_start = time.perf_counter()

            n = len(points)

            if n <= k:
                k = n - 1

            k_actual = k + 1

            upload_time = 0.0
            compute_time = 0.0
            download_time = 0.0
            finalize_time = 0.0

            peak_reserved_pool_mb = 0.0
            peak_used_pool_mb = 0.0

            # Загрузка точек на GPU
            t0 = time.perf_counter()
            points_gpu = self.cp.asarray(points, dtype=self.cp.float32)
            points_sq = self.cp.sum(points_gpu * points_gpu, axis=1)
            self._sync()
            upload_time += time.perf_counter() - t0

            mem_info = self._get_gpu_mem_info()
            if mem_info:
                peak_reserved_pool_mb = max(peak_reserved_pool_mb, mem_info["reserved_pool_mb"])
                peak_used_pool_mb = max(peak_used_pool_mb, mem_info["used_pool_mb"])

            all_indices = np.empty((n, k), dtype=np.int32)
            all_distances = np.empty((n, k), dtype=np.float32)

            # Основной цикл по query-батчам
            for q_start in range(0, n, self.query_batch_size):
                q_end = min(q_start + self.query_batch_size, n)

                t0 = time.perf_counter()
                query = points_gpu[q_start:q_end]
                query_sq = self.cp.sum(query * query, axis=1)
                self._sync()
                compute_time += time.perf_counter() - t0

                q_size = q_end - q_start

                top_dist = self.cp.full(
                    (q_size, k_actual),
                    self.cp.inf,
                    dtype=self.cp.float32,
                )

                top_idx = self.cp.full(
                    (q_size, k_actual),
                    -1,
                    dtype=self.cp.int32,
                )

                # Внутренний цикл по ref-батчам
                for r_start in range(0, n, self.ref_batch_size):
                    r_end = min(r_start + self.ref_batch_size, n)

                    t0 = time.perf_counter()

                    ref = points_gpu[r_start:r_end]
                    ref_sq = points_sq[r_start:r_end]

                    dist = (
                        query_sq[:, None]
                        + ref_sq[None, :]
                        - 2.0 * query @ ref.T
                    )

                    self.cp.maximum(dist, 0.0, out=dist)

                    global_idx = self.cp.arange(r_start, r_end, dtype=self.cp.int32)
                    expanded_idx = self.cp.broadcast_to(global_idx, dist.shape)

                    combined_dist = self.cp.concatenate((top_dist, dist), axis=1)
                    combined_idx = self.cp.concatenate((top_idx, expanded_idx), axis=1)

                    idx_part = self.cp.argpartition(combined_dist, k_actual, axis=1)[:, :k_actual]
                    row_ids = self.cp.arange(q_size)[:, None]

                    top_dist = combined_dist[row_ids, idx_part]
                    top_idx = combined_idx[row_ids, idx_part]

                    self._sync()
                    compute_time += time.perf_counter() - t0

                    del ref, ref_sq, dist, global_idx, expanded_idx, combined_dist, combined_idx, idx_part, row_ids

                    if self.cleanup_each_ref_batch:
                        self.cp.get_default_memory_pool().free_all_blocks()

                    mem_info = self._get_gpu_mem_info()
                    if mem_info:
                        peak_reserved_pool_mb = max(peak_reserved_pool_mb, mem_info["reserved_pool_mb"])
                        peak_used_pool_mb = max(peak_used_pool_mb, mem_info["used_pool_mb"])

                # Финальная сортировка top-k
                t0 = time.perf_counter()
                order = self.cp.argsort(top_dist, axis=1)

                top_dist = self.cp.take_along_axis(top_dist, order, axis=1)
                top_idx = self.cp.take_along_axis(top_idx, order, axis=1)

                top_dist = top_dist[:, 1:]
                top_idx = top_idx[:, 1:]

                if self.use_sqrt:
                    self.cp.sqrt(top_dist, out=top_dist)

                self._sync()
                finalize_time += time.perf_counter() - t0

                # Копирование результата обратно на CPU
                #
                t0 = time.perf_counter()
                all_indices[q_start:q_end] = self.cp.asnumpy(top_idx)
                all_distances[q_start:q_end] = self.cp.asnumpy(top_dist)
                self._sync()
                download_time += time.perf_counter() - t0

                del query, query_sq, top_dist, top_idx, order

                if self.cleanup_each_query_batch:
                    self.cp.get_default_memory_pool().free_all_blocks()

                mem_info = self._get_gpu_mem_info()
                if mem_info:
                    peak_reserved_pool_mb = max(peak_reserved_pool_mb, mem_info["reserved_pool_mb"])
                    peak_used_pool_mb = max(peak_used_pool_mb, mem_info["used_pool_mb"])

            # -------------------------------
            # Формирование результата на CPU
            # -------------------------------
            t0 = time.perf_counter()
            neighbors_dict = {
                i: {
                    "indices": all_indices[i].tolist(),
                    "distances": all_distances[i].tolist(),
                }
                for i in range(n)
            }
            finalize_time += time.perf_counter() - t0

            self.clear_cache()

            total_time = time.perf_counter() - t_total_start

            if self.log_timing:
                logger.info(
                    "GPU KNN timing: "
                    f"upload={upload_time:.3f}s, "
                    f"compute={compute_time:.3f}s, "
                    f"download={download_time:.3f}s, "
                    f"finalize={finalize_time:.3f}s, "
                    f"total={total_time:.3f}s"
                )
                logger.info(
                    "GPU memory usage: "
                    f"peak_used_pool={peak_used_pool_mb:.1f} MB, "
                    f"peak_reserved_pool={peak_reserved_pool_mb:.1f} MB"
                )

            return neighbors_dict

        except Exception as e:
            logger.error(f"GPU KNN ошибка: {e}")
            self.clear_cache()
            return None