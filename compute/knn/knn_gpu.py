import logging
import math
import time
from functools import lru_cache

import cupy as cp
import numpy as np

from compute.backend_policy import GPUUnavailableError, classify_gpu_exception
from compute.knn.result_format import make_compact_neighbors

logger = logging.getLogger("WaveletApp")


# KNN feature coordinates in the studio are image pixel coordinates.  The
# point pipeline normalizes them to Nx2 int32 before KNN, then knn_cpu.py casts
# them to float32 for the common CPU/GPU API.  The optimized GPU path below
# detects that integer-valued contract and uses an exact 2-D spatial grid.
# This avoids the old O(N^2) all-pairs distance scan while keeping the whole
# neighbour search on CUDA.
_GRID_MAX_K = 32
_GRID_MIN_POINTS = 2048
_GRID_TARGET_POINTS_PER_CELL = 48


def _grid_kernel_source(k: int) -> str:
    """CUDA source for exact KNN on integer 2-D image coordinates.

    One CUDA thread owns one query point.  Points are indexed by a uniform
    spatial grid.  The thread expands cell rings until the k-th current
    neighbour is strictly closer than every point that can still exist outside
    the searched rectangle.  This stopping criterion makes the result exact,
    not approximate.

    Squared distances are accumulated in signed 64-bit integers, so the
    ranking is free from the float32 cancellation that affected the historical
    ||q||^2 + ||r||^2 - 2 q*r implementation.
    """
    return rf'''
#define K {int(k)}
#define INF_D2 0x7fffffffffffffffLL

__device__ __forceinline__ bool better_pair(
        long long d2, int idx, long long old_d2, int old_idx) {{
    return (d2 < old_d2) || (d2 == old_d2 && (old_idx < 0 || idx < old_idx));
}}

__device__ __forceinline__ void consider_candidate(
        int cand,
        int qid,
        int qx,
        int qy,
        const int* points,
        long long* best_d2,
        int* best_idx) {{
    if (cand == qid) return;

    const int cx = points[2 * cand];
    const int cy = points[2 * cand + 1];
    const long long dx = (long long)cx - (long long)qx;
    const long long dy = (long long)cy - (long long)qy;
    const long long d2 = dx * dx + dy * dy;

    if (!better_pair(d2, cand, best_d2[K - 1], best_idx[K - 1])) return;

    int pos = K - 1;
    while (pos > 0 && better_pair(d2, cand, best_d2[pos - 1], best_idx[pos - 1])) {{
        best_d2[pos] = best_d2[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
    }}
    best_d2[pos] = d2;
    best_idx[pos] = cand;
}}

__device__ __forceinline__ void scan_cell(
        int gx,
        int gy,
        int grid_w,
        int grid_h,
        const int* cell_starts,
        const int* sorted_indices,
        const int* points,
        int qid,
        int qx,
        int qy,
        long long* best_d2,
        int* best_idx) {{
    if (gx < 0 || gy < 0 || gx >= grid_w || gy >= grid_h) return;
    const int cell = gy * grid_w + gx;
    const int begin = cell_starts[cell];
    const int end = cell_starts[cell + 1];
    for (int p = begin; p < end; ++p) {{
        consider_candidate(
            sorted_indices[p], qid, qx, qy, points, best_d2, best_idx
        );
    }}
}}

extern "C" __global__ void knn_grid_exact_i32(
        const int* points,
        const int n,
        const int* sorted_indices,
        const int* cell_starts,
        const int min_x,
        const int min_y,
        const int cell_size,
        const int grid_w,
        const int grid_h,
        int* out_indices,
        float* out_distances,
        float* out_angles) {{
    const int qid = blockDim.x * blockIdx.x + threadIdx.x;
    if (qid >= n) return;

    const int qx = points[2 * qid];
    const int qy = points[2 * qid + 1];
    const int qcx = (qx - min_x) / cell_size;
    const int qcy = (qy - min_y) / cell_size;

    long long best_d2[K];
    int best_idx[K];
    #pragma unroll
    for (int j = 0; j < K; ++j) {{
        best_d2[j] = INF_D2;
        best_idx[j] = -1;
    }}

    int max_ring_x = qcx;
    if (grid_w - 1 - qcx > max_ring_x) max_ring_x = grid_w - 1 - qcx;
    int max_ring_y = qcy;
    if (grid_h - 1 - qcy > max_ring_y) max_ring_y = grid_h - 1 - qcy;
    const int max_ring = max_ring_x > max_ring_y ? max_ring_x : max_ring_y;

    for (int ring = 0; ring <= max_ring; ++ring) {{
        if (ring == 0) {{
            scan_cell(
                qcx, qcy, grid_w, grid_h, cell_starts, sorted_indices,
                points, qid, qx, qy, best_d2, best_idx
            );
        }} else {{
            const int x0 = qcx - ring;
            const int x1 = qcx + ring;
            const int y0 = qcy - ring;
            const int y1 = qcy + ring;

            // Top/bottom edges of the square ring.
            for (int gx = x0; gx <= x1; ++gx) {{
                scan_cell(
                    gx, y0, grid_w, grid_h, cell_starts, sorted_indices,
                    points, qid, qx, qy, best_d2, best_idx
                );
                if (y1 != y0) {{
                    scan_cell(
                        gx, y1, grid_w, grid_h, cell_starts, sorted_indices,
                        points, qid, qx, qy, best_d2, best_idx
                    );
                }}
            }}
            // Left/right edges without the already scanned corners.
            for (int gy = y0 + 1; gy <= y1 - 1; ++gy) {{
                scan_cell(
                    x0, gy, grid_w, grid_h, cell_starts, sorted_indices,
                    points, qid, qx, qy, best_d2, best_idx
                );
                if (x1 != x0) {{
                    scan_cell(
                        x1, gy, grid_w, grid_h, cell_starts, sorted_indices,
                        points, qid, qx, qy, best_d2, best_idx
                    );
                }}
            }}
        }}

        if (best_idx[K - 1] < 0) continue;

        const int sx0 = qcx - ring > 0 ? qcx - ring : 0;
        const int sx1 = qcx + ring < grid_w - 1 ? qcx + ring : grid_w - 1;
        const int sy0 = qcy - ring > 0 ? qcy - ring : 0;
        const int sy1 = qcy + ring < grid_h - 1 ? qcy + ring : grid_h - 1;

        // Exact lower bound to every still-unsearched integer coordinate.
        // We stop only on strict inequality.  If an unseen point can tie the
        // current k-th distance, another ring is visited so deterministic
        // (distance, global-index) tie breaking remains exact as well.
        long long unseen_bound2 = INF_D2;

        if (sx0 > 0) {{
            const long long boundary = (long long)min_x + (long long)sx0 * cell_size;
            const long long d = (long long)qx - (boundary - 1LL);
            const long long d2 = d * d;
            if (d2 < unseen_bound2) unseen_bound2 = d2;
        }}
        if (sx1 < grid_w - 1) {{
            const long long boundary = (long long)min_x + (long long)(sx1 + 1) * cell_size;
            const long long d = boundary - (long long)qx;
            const long long d2 = d * d;
            if (d2 < unseen_bound2) unseen_bound2 = d2;
        }}
        if (sy0 > 0) {{
            const long long boundary = (long long)min_y + (long long)sy0 * cell_size;
            const long long d = (long long)qy - (boundary - 1LL);
            const long long d2 = d * d;
            if (d2 < unseen_bound2) unseen_bound2 = d2;
        }}
        if (sy1 < grid_h - 1) {{
            const long long boundary = (long long)min_y + (long long)(sy1 + 1) * cell_size;
            const long long d = boundary - (long long)qy;
            const long long d2 = d * d;
            if (d2 < unseen_bound2) unseen_bound2 = d2;
        }}

        if (unseen_bound2 == INF_D2 || best_d2[K - 1] < unseen_bound2) break;
    }}

    #pragma unroll
    for (int j = 0; j < K; ++j) {{
        const int out_pos = qid * K + j;
        const int idx = best_idx[j];
        out_indices[out_pos] = idx;
        if (idx >= 0) {{
            out_distances[out_pos] = sqrtf((float)best_d2[j]);
            const int tx = points[2 * idx];
            const int ty = points[2 * idx + 1];
            const int dx = tx - qx;
            const int dy = ty - qy;
            if (dx == 0 && dy == 0) {{
                out_angles[out_pos] = __int_as_float(0x7fffffff);
            }} else {{
                float angle = atan2f((float)dx, (float)(-dy))
                    * 57.2957795130823208768f;
                if (angle < 0.0f) angle += 360.0f;
                out_angles[out_pos] = angle;
            }}
        }} else {{
            out_distances[out_pos] = __int_as_float(0x7f800000);
            out_angles[out_pos] = __int_as_float(0x7fffffff);
        }}
    }}
}}
'''


@lru_cache(maxsize=_GRID_MAX_K)
def _compiled_grid_kernel(k: int):
    return cp.RawKernel(
        _grid_kernel_source(int(k)),
        "knn_grid_exact_i32",
        options=("--std=c++11",),
    )


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
        self.last_algorithm = None
        self.last_details = {}
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            test = self.cp.array([1.0, 2.0, 3.0], dtype=self.cp.float32)
            self.cp.sum(test)
            self.cp.cuda.Stream.null.synchronize()
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
        try:
            self.cp.cuda.Stream.null.synchronize()
        except Exception as e:
            logger.error(f"{e}")

    def _get_gpu_mem_info(self):
        try:
            free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
            pool = self.cp.get_default_memory_pool()
            return {
                "free_mb": free_bytes / (1024 ** 2),
                "total_mb": total_bytes / (1024 ** 2),
                "used_pool_mb": pool.used_bytes() / (1024 ** 2),
                "reserved_pool_mb": pool.total_bytes() / (1024 ** 2),
            }
        except Exception:
            return None

    @staticmethod
    def _integer_image_points(points: np.ndarray) -> np.ndarray | None:
        """Return int32 coordinates when the KNN cloud is pixel-aligned."""
        arr = np.asarray(points)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            return None
        if not np.all(np.isfinite(arr)):
            return None
        rounded = np.rint(arr)
        if not np.all(np.abs(arr - rounded) <= 1e-5):
            return None
        i64 = rounded.astype(np.int64, copy=False)
        if i64.min() < np.iinfo(np.int32).min or i64.max() > np.iinfo(np.int32).max:
            return None
        return np.ascontiguousarray(i64, dtype=np.int32)

    @staticmethod
    def _choose_cell_size(points_i32: np.ndarray) -> tuple[int, int, int, int, int]:
        min_x = int(points_i32[:, 0].min())
        max_x = int(points_i32[:, 0].max())
        min_y = int(points_i32[:, 1].min())
        max_y = int(points_i32[:, 1].max())
        span_x = max(1, max_x - min_x + 1)
        span_y = max(1, max_y - min_y + 1)
        n = max(1, len(points_i32))
        area = max(1, span_x * span_y)
        suggested = int(round(math.sqrt(_GRID_TARGET_POINTS_PER_CELL * area / n)))
        cell_size = max(1, min(128, suggested))
        grid_w = (span_x + cell_size - 1) // cell_size
        grid_h = (span_y + cell_size - 1) // cell_size
        return min_x, min_y, cell_size, grid_w, grid_h

    def _find_grid_exact(self, points_i32: np.ndarray, k: int):
        """Exact CUDA KNN using a spatial grid for 2-D integer image points."""
        t_total = time.perf_counter()
        n = len(points_i32)
        min_x, min_y, cell_size, grid_w, grid_h = self._choose_cell_size(points_i32)
        num_cells = int(grid_w * grid_h)

        t0 = time.perf_counter()
        points_gpu = self.cp.asarray(points_i32, dtype=self.cp.int32)
        px = points_gpu[:, 0]
        py = points_gpu[:, 1]
        cell_x = (px - np.int32(min_x)) // np.int32(cell_size)
        cell_y = (py - np.int32(min_y)) // np.int32(cell_size)
        cell_ids = (cell_y * np.int32(grid_w) + cell_x).astype(self.cp.int32, copy=False)

        order = self.cp.argsort(cell_ids).astype(self.cp.int32, copy=False)
        counts = self.cp.bincount(cell_ids, minlength=num_cells).astype(self.cp.int32, copy=False)
        cell_starts = self.cp.empty(num_cells + 1, dtype=self.cp.int32)
        cell_starts[0] = 0
        self.cp.cumsum(counts, dtype=self.cp.int32, out=cell_starts[1:])
        out_indices = self.cp.empty((n, k), dtype=self.cp.int32)
        out_distances = self.cp.empty((n, k), dtype=self.cp.float32)
        out_angles = self.cp.empty((n, k), dtype=self.cp.float32)
        self._sync()
        upload_and_index = time.perf_counter() - t0

        t0 = time.perf_counter()
        kernel = _compiled_grid_kernel(k)
        threads = 128
        blocks = (n + threads - 1) // threads
        kernel(
            (blocks,), (threads,),
            (
                points_gpu, np.int32(n), order, cell_starts,
                np.int32(min_x), np.int32(min_y), np.int32(cell_size),
                np.int32(grid_w), np.int32(grid_h),
                out_indices, out_distances, out_angles,
            ),
        )
        self._sync()
        compute_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        indices = self.cp.asnumpy(out_indices)
        distances = self.cp.asnumpy(out_distances)
        angles = self.cp.asnumpy(out_angles)
        self._sync()
        download_time = time.perf_counter() - t0

        compact = make_compact_neighbors(indices, distances, angles)
        total = time.perf_counter() - t_total

        mem_info = self._get_gpu_mem_info() or {}
        self.last_algorithm = "cuda-spatial-grid-exact-i32-v1"
        self.last_details = {
            "algorithm": self.last_algorithm,
            "points": int(n),
            "k": int(k),
            "cell_size": int(cell_size),
            "grid_width": int(grid_w),
            "grid_height": int(grid_h),
            "grid_cells": int(num_cells),
            "exact": True,
            "distance_accumulator": "int64-squared-euclidean",
            "angles_backend": "cuda-fused",
            "angle_definition": "from_vertical_up_clockwise",
        }

        if self.log_timing:
            logger.info(
                "GPU KNN grid timing: "
                f"index={upload_and_index:.3f}s, "
                f"compute={compute_time:.3f}s, "
                f"download={download_time:.3f}s, "
                f"total={total:.3f}s, "
                f"cell={cell_size}px, grid={grid_w}x{grid_h}"
            )
            if mem_info:
                logger.info(
                    "GPU memory usage: "
                    f"used_pool={mem_info.get('used_pool_mb', 0.0):.1f} MB, "
                    f"reserved_pool={mem_info.get('reserved_pool_mb', 0.0):.1f} MB"
                )

        del points_gpu, px, py, cell_x, cell_y, cell_ids, order, counts
        del cell_starts, out_indices, out_distances, out_angles
        self.clear_cache()
        return compact

    def _find_pairwise_batched(self, points: np.ndarray, k: int):
        """Compatibility path for non-pixel point clouds.

        This retains the numerically stable direct-difference implementation,
        but production image KNN should normally use the spatial-grid path.
        """
        t_total_start = time.perf_counter()
        n = len(points)
        upload_time = compute_time = download_time = finalize_time = 0.0
        peak_reserved_pool_mb = peak_used_pool_mb = 0.0

        t0 = time.perf_counter()
        points_gpu = self.cp.asarray(points, dtype=self.cp.float32)
        self._sync()
        upload_time += time.perf_counter() - t0

        all_indices = np.empty((n, k), dtype=np.int32)
        all_distances = np.empty((n, k), dtype=np.float32)

        for q_start in range(0, n, self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, n)
            query = points_gpu[q_start:q_end]
            q_size = q_end - q_start
            top_dist = self.cp.full((q_size, k), self.cp.inf, dtype=self.cp.float32)
            top_idx = self.cp.full((q_size, k), -1, dtype=self.cp.int32)

            for r_start in range(0, n, self.ref_batch_size):
                r_end = min(r_start + self.ref_batch_size, n)
                t0 = time.perf_counter()
                ref = points_gpu[r_start:r_end]
                dx = query[:, 0, None] - ref[None, :, 0]
                dy = query[:, 1, None] - ref[None, :, 1]
                dist = dx * dx
                dist += dy * dy
                del dx, dy

                overlap_start = max(q_start, r_start)
                overlap_end = min(q_end, r_end)
                if overlap_start < overlap_end:
                    global_self = self.cp.arange(overlap_start, overlap_end, dtype=self.cp.int32)
                    dist[global_self - q_start, global_self - r_start] = self.cp.inf

                global_idx = self.cp.arange(r_start, r_end, dtype=self.cp.int32)
                expanded_idx = self.cp.broadcast_to(global_idx, dist.shape)
                combined_dist = self.cp.concatenate((top_dist, dist), axis=1)
                combined_idx = self.cp.concatenate((top_idx, expanded_idx), axis=1)
                idx_part = self.cp.argpartition(combined_dist, k, axis=1)[:, :k]
                row_ids = self.cp.arange(q_size)[:, None]
                top_dist = combined_dist[row_ids, idx_part]
                top_idx = combined_idx[row_ids, idx_part]
                self._sync()
                compute_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            order = self.cp.argsort(top_dist, axis=1)
            top_dist = self.cp.take_along_axis(top_dist, order, axis=1)
            top_idx = self.cp.take_along_axis(top_idx, order, axis=1)
            if self.use_sqrt:
                self.cp.sqrt(top_dist, out=top_dist)
            self._sync()
            finalize_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            all_indices[q_start:q_end] = self.cp.asnumpy(top_idx)
            all_distances[q_start:q_end] = self.cp.asnumpy(top_dist)
            self._sync()
            download_time += time.perf_counter() - t0

            mem_info = self._get_gpu_mem_info()
            if mem_info:
                peak_reserved_pool_mb = max(peak_reserved_pool_mb, mem_info["reserved_pool_mb"])
                peak_used_pool_mb = max(peak_used_pool_mb, mem_info["used_pool_mb"])

        compact = make_compact_neighbors(all_indices, all_distances)
        total_time = time.perf_counter() - t_total_start
        self.last_algorithm = "cuda-pairwise-batched-direct-v2"
        self.last_details = {"algorithm": self.last_algorithm, "exact": True}
        self.clear_cache()

        if self.log_timing:
            logger.info(
                "GPU KNN pairwise timing: "
                f"upload={upload_time:.3f}s, compute={compute_time:.3f}s, "
                f"download={download_time:.3f}s, finalize={finalize_time:.3f}s, "
                f"total={total_time:.3f}s"
            )
            logger.info(
                "GPU memory usage: "
                f"peak_used_pool={peak_used_pool_mb:.1f} MB, "
                f"peak_reserved_pool={peak_reserved_pool_mb:.1f} MB"
            )
        return compact

    def find_k_nearest_neighbors(self, points: np.ndarray, k: int):
        if not self._available:
            raise GPUUnavailableError("GPU KNN processor is unavailable")
        if len(points) < 2:
            return {}

        try:
            points = np.asarray(points)
            n = len(points)
            if n <= k:
                k = n - 1
            if k <= 0:
                return {}

            integer_points = self._integer_image_points(points)
            if (
                integer_points is not None
                and n >= _GRID_MIN_POINTS
                and k <= _GRID_MAX_K
            ):
                return self._find_grid_exact(integer_points, int(k))

            # Small clouds do not suffer from quadratic scaling and the old
            # batched path avoids one-time NVRTC/grid-index overhead.  It also
            # remains the compatibility path for non-integer feature spaces.
            return self._find_pairwise_batched(points, int(k))

        except Exception as e:
            self.clear_cache()
            mapped = classify_gpu_exception(e)
            if mapped is None:
                logger.exception("GPU KNN algorithm error")
                raise
            logger.error(f"GPU KNN infrastructure error: {mapped}")
            raise mapped from e
