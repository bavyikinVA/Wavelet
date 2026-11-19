"""Optional ROI point branch.  Legacy unmasked processing never calls here.

CWT comes from the original full image.  Each region is then processed as an
independent collection of connected row/column intervals.  Overlapping pixels
are computed once per region, never reassigned to a global exclusive label.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from compute.roi_analysis import masked_extrema, masked_envelopes
from compute.knn.knn_cpu import process_extremes_with_knn
from result_naming import point_stem, scale_folder_name
from history.atomic_io import atomic_savez, atomic_text_writer


def process_roi_points(processor, task, *, plan, row_var, col_var, max_var,
                       min_var, k, knn_text, knn_image):
    """Calculate/save ROI-specific extrema, envelopes, KNN, and point counts.

    Full-size results are kept on disk in per-ROI folders.  In-memory KNN
    dataset keys encode their region in the channel token, e.g. r@roi2, while
    source data uses the ordinary channel and untouched full-image CWT.
    """
    regions = task.roi_collection
    if regions is None or len(regions) == 0:
        raise ValueError("ROI-конвейер вызван без выделенных областей")
    if task.analysis_mode != "1d":
        raise ValueError("ROI-этапы после CWT пока поддерживаются только для 1D")
    if not (row_var or col_var):
        raise ValueError("Выберите хотя бы одно направление CWT")
    if getattr(plan, "synchronization", False):
        raise ValueError("Межстрочная синхронизация для независимых ROI пока не реализована. "
                         "Отключите её или выполните анализ без ROI.")
    if not (max_var or min_var):
        max_var = True
    protocol = getattr(task, "execution_protocol", None)
    analysis_axes = ("row", "col")
    cwt_axes = [(0, "row")] if row_var else []
    if col_var:
        cwt_axes.append((1, "col"))
    codes = list(task.analysis_channel_codes)
    if not codes:
        codes = list(task.analysis_channel_codes_for_settings())
    operation_count = sum(len(task.result[axis]) * task.num_scale * len(regions)
                          for axis, _ in cwt_axes)
    operation_index = 0
    statistics = []
    task.roi_knn_results = {}
    for axis_index, cwt_axis in cwt_axes:
        for channel_index, code in enumerate(codes):
            for scale_index, scale in enumerate(task.scales):
                coeffs = np.asarray(task.result[axis_index][channel_index][scale_index])
                if coeffs.shape != regions.shape:
                    raise ValueError("Размер CWT и ROI несовместим")
                for region in regions.regions:
                    operation_index += 1
                    check = getattr(getattr(processor.progress, "run_control", None), "check", None)
                    if callable(check):
                        check()
                    processor.progress.update_progress(
                        0.1 + 0.8 * operation_index / max(operation_count, 1),
                        f"ROI {region.id}/{len(regions)} · {code} · a={scale:g}"
                    )
                    roi_root = (Path(task.task_folder_path) / "roi" /
                                f"region_{region.id:04d}" / "scales" /
                                scale_folder_name(scale))
                    roi_root.mkdir(parents=True, exist_ok=True)
                    pmaxr, pmaxc, pminr, pminc = masked_extrema(
                        coeffs, region.mask,
                        distance=task.extrema_distance,
                        prominence=task.extrema_prominence,
                        max_var=max_var, min_var=min_var,
                    )
                    pairs = {
                        "row": (pmaxr, pminr), "col": (pmaxc, pminc),
                    }
                    knn_inputs = {"type_data": axis_index, "cwt_axis": cwt_axis,
                                  "channel": code, "scale": scale,
                                  "max_by_row": [], "max_by_column": [],
                                  "min_by_row": [], "min_by_column": []}
                    for feature_axis in analysis_axes:
                        maxima, minima = pairs[feature_axis]
                        if plan.envelopes:
                            upper, lower = masked_envelopes(
                                coeffs, region.mask, maxima, minima,
                                direction=feature_axis,
                            )
                        else:
                            upper = np.empty((0, 2), dtype=np.int32)
                            lower = np.empty((0, 2), dtype=np.int32)
                        for kind, points, is_envelope in (
                            ("max", maxima, False), ("min", minima, False),
                            ("upper", upper, True), ("lower", lower, True),
                        ):
                            if (kind in ("max", "upper") and not max_var or
                                kind in ("min", "lower") and not min_var or
                                is_envelope and not plan.envelopes):
                                continue
                            artifact = "env" if is_envelope else "ext"
                            folder = roi_root / ("envelopes" if is_envelope else "extremes")
                            folder.mkdir(parents=True, exist_ok=True)
                            stem = point_stem(artifact, cwt_axis, feature_axis, code, scale, kind)
                            # Save empty layers as well, so they are distinguishable
                            # from missing/failed calculations on disk.
                            atomic_savez(folder / (stem + ".npz"),
                                         points=np.asarray(points, dtype=np.int32).reshape(-1, 2),
                                         shape=np.asarray(coeffs.shape, dtype=np.int32),
                                         roi_id=np.int32(region.id))
                            save_text = (task.output_envelopes_text if is_envelope
                                         else task.output_extremes_text)
                            save_image = (task.output_envelopes_image if is_envelope
                                          else task.output_extremes_image)
                            if save_text:
                                processor.save_extremes_to_file(str(folder), stem, points)
                            if save_image and len(points):
                                processor.save_extremes_graphic(
                                    str(folder), stem, points, coefs_2d=coeffs)
                        suffix = "row" if feature_axis == "row" else "column"
                        if max_var:
                            knn_inputs[f"max_by_{suffix}"] = upper if plan.envelopes else maxima
                        if min_var:
                            knn_inputs[f"min_by_{suffix}"] = lower if plan.envelopes else minima
                        statistics.append({
                            "roi_id": region.id, "cwt_axis": cwt_axis,
                            "feature_axis": feature_axis, "channel": code,
                            "scale": float(scale), "maxima": len(maxima),
                            "minima": len(minima), "upper": len(upper),
                            "lower": len(lower),
                        })
                    if plan.knn:
                        knn_dir = roi_root / "knn"
                        knn_dir.mkdir(exist_ok=True)
                        results = process_extremes_with_knn(
                            knn_inputs, str(knn_dir), k, task.original_image,
                            knn_text, knn_image, use_gpu=processor.backend.use_gpu,
                            source_direction=cwt_axis,
                            log_callback=processor.progress.log_info,
                            protocol=protocol, print_npz_var=True,
                            png_max_points=int(getattr(task, "knn_png_max_points", 50000)),
                        )
                        for point_type, payload in results.items():
                            key = (cwt_axis, f"{code}@roi{region.id}",
                                   float(scale), point_type)
                            task.knn_results[key] = payload
                            task.roi_knn_results[(region.id, cwt_axis, code,
                                                  float(scale), point_type)] = payload
    if statistics:
        path = Path(task.task_folder_path) / "roi" / "statistics.csv"
        with atomic_text_writer(path, encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
            writer.writeheader()
            writer.writerows(statistics)
        if plan.statistics:
            task.statistics_results = {"roi_statistics": str(path)}
    processor.progress.log_info(
        f"ROI: рассчитано {len(regions)} независимых областей; "
        f"перекрывающиеся пиксели учитываются в каждой ROI отдельно"
    )
    processor.progress.update_progress(1.0, "Анализ ROI завершён")
