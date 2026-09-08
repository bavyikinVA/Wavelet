import cupy as cp
from cupyx.scipy.signal import find_peaks


class GPUEnvelopeProcessor:
    def __init__(self, max_gpu_memory_mb=6000):
        """Инициализация с ограничением по памяти GPU"""
        self.max_gpu_memory = max_gpu_memory_mb * 1024 * 1024  # байты
        self.device_memory = cp.cuda.Device().mem_info
        print(f"Доступно GPU памяти: {self.device_memory[0] / 1e9:.2f} GB")

    def get_row_envelopes_gpu(self, coefs, max_points, min_points):
        """
        GPU-реализация построения огибающих для строк
        с batch-обработкой для избежания переполнения памяти
        """
        # Преобразуем ВСЕ входные данные на GPU единовременно
        coefs_gpu = cp.asarray(coefs, dtype=cp.float32)  # Используем float32 для экономии памяти
        rows, cols = coefs_gpu.shape

        # Создаем бинарные маски максимумов и минимумов на GPU
        max_mask_gpu = cp.zeros((rows, cols), dtype=cp.bool_)
        min_mask_gpu = cp.zeros((rows, cols), dtype=cp.bool_)

        if max_points:
            # Векторизованное создание маски максимумов
            max_points_gpu = cp.array(max_points, dtype=cp.int32)
            max_mask_gpu[max_points_gpu[:, 1], max_points_gpu[:, 0]] = True

        if min_points:
            # Векторизованное создание маски минимумов
            min_points_gpu = cp.array(min_points, dtype=cp.int32)
            min_mask_gpu[min_points_gpu[:, 1], min_points_gpu[:, 0]] = True

        # Определяем оптимальный batch size на основе доступной памяти
        memory_per_row = cols * 4 * 5  # float32 * 5 временных массивов
        max_batch_size = min(rows, int(self.max_gpu_memory * 0.8 / memory_per_row))
        max_batch_size = max(1, max_batch_size // 32 * 32)  # Выравниваем для CUDA
        print(f"Batch size: {max_batch_size} строк из {rows}")

        # Подготавливаем результаты
        all_upper_peaks = []
        all_lower_peaks = []

        # Batch-обработка строк
        for batch_start in range(0, rows, max_batch_size):
            batch_end = min(batch_start + max_batch_size, rows)
            batch_rows = batch_end - batch_start

            # Выделяем память под batch на GPU
            batch_coefs = coefs_gpu[batch_start:batch_end, :]
            batch_max_mask = max_mask_gpu[batch_start:batch_end, :]
            batch_min_mask = min_mask_gpu[batch_start:batch_end, :]

            # Векторизованная интерполяция для ВСЕХ строк в batch одновременно
            batch_upper_env = self._interpolate_batch_gpu(
                batch_coefs, batch_max_mask, cols
            )
            batch_lower_env = self._interpolate_batch_gpu(
                batch_coefs, batch_min_mask, cols
            )

            # Векторизованный поиск пиков для ВСЕХ строк в batch
            batch_upper_peaks = self._find_peaks_batch_gpu(batch_upper_env, direction='max')
            batch_lower_peaks = self._find_peaks_batch_gpu(batch_lower_env, direction='min')

            # Преобразуем результаты в координаты
            for i in range(batch_rows):
                actual_row = batch_start + i
                upper_peaks = batch_upper_peaks[i]
                lower_peaks = batch_lower_peaks[i]

                if len(upper_peaks) > 0:
                    all_upper_peaks.extend([(int(x), actual_row) for x in upper_peaks])
                if len(lower_peaks) > 0:
                    all_lower_peaks.extend([(int(x), actual_row) for x in lower_peaks])

            # Очищаем память GPU от временных массивов batch
            del batch_coefs, batch_max_mask, batch_min_mask
            del batch_upper_env, batch_lower_env
            del batch_upper_peaks, batch_lower_peaks
            cp.get_default_memory_pool().free_all_blocks()

        return all_upper_peaks, all_lower_peaks

    @staticmethod
    def _interpolate_batch_gpu(batch_coefs, batch_mask, cols):
        """
        Векторизованная интерполяция для batch строк
        batch_coefs: (batch_size, cols)
        batch_mask: (batch_size, cols) bool
        Возвращает: (batch_size, cols) интерполированные значения
        """
        batch_size = batch_coefs.shape[0]

        # 1. Создаем равномерную сетку для всех строк
        t = cp.arange(cols, dtype=cp.float32)

        # Missing support is undefined; never substitute the original signal.
        result = cp.full(batch_coefs.shape, cp.nan, dtype=cp.float32)

        # 3. Для каждой строки в batch выполняем интерполяцию
        for i in range(batch_size):
            # Получаем маску для текущей строки
            row_mask = batch_mask[i]

            # Проверяем, достаточно ли точек для интерполяции
            valid_points = cp.sum(row_mask)
            if valid_points >= 1:
                # Получаем координаты и значения
                points_idx = cp.where(row_mask)[0]
                values = batch_coefs[i, points_idx]

                # Сортируем
                sort_idx = cp.argsort(points_idx)
                points_idx_sorted = points_idx[sort_idx]
                values_sorted = values[sort_idx]

                # Линейная интерполяция
                interpolated = cp.interp(t, points_idx_sorted.astype(cp.float32), values_sorted)
                result[i] = interpolated

        return result

    @staticmethod
    def _find_peaks_batch_gpu(batch_data, direction='max'):
        """
        Векторизованный поиск пиков для batch данных
        Остается полностью на GPU
        """
        batch_size = batch_data.shape[0]
        all_peaks = []

        for i in range(batch_size):
            row_data = batch_data[i]

            if direction == 'max':
                peaks, _ = find_peaks(row_data)
            else:  # 'min'
                peaks, _ = find_peaks(-row_data)

            all_peaks.append(peaks.get() if hasattr(peaks, 'get') else peaks)

        return all_peaks

    def get_col_envelopes_gpu(self, coefs, max_points, min_points):
        """
        GPU-реализация построения огибающих для столбцов
        с batch-обработкой для избежания переполнения памяти
        """
        # Преобразуем ВСЕ входные данные на GPU единовременно
        coefs_gpu = cp.asarray(coefs, dtype=cp.float32)  # Используем float32 для экономии памяти
        rows, cols = coefs_gpu.shape

        # Транспонируем данные для обработки столбцов как строк
        # Теперь работаем с (cols, rows) вместо (rows, cols)
        coefs_transposed = coefs_gpu.T

        # Создаем бинарные маски максимумов и минимумов на GPU
        # Для столбцов точки имеют формат (x, y), где x - номер столбца, y - номер строки
        max_mask_gpu = cp.zeros((cols, rows), dtype=cp.bool_)  # Транспонированный размер
        min_mask_gpu = cp.zeros((cols, rows), dtype=cp.bool_)  # Транспонированный размер

        if max_points:
            # Для столбцов: точки в формате (col, row), но для транспонированной матрицы
            # нам нужно инвертировать координаты: (row, col) -> (col, row)
            max_points_gpu = cp.array(max_points, dtype=cp.int32)
            # max_points_gpu[:, 0] - координата x (столбец)
            # max_points_gpu[:, 1] - координата y (строка)
            # Для транспонированной матрицы индексы: [col, row]
            max_mask_gpu[max_points_gpu[:, 0], max_points_gpu[:, 1]] = True

        if min_points:
            min_points_gpu = cp.array(min_points, dtype=cp.int32)
            min_mask_gpu[min_points_gpu[:, 0], min_points_gpu[:, 1]] = True

        # Определяем оптимальный batch size для столбцов
        memory_per_col = rows * 4 * 5  # float32 * 5 временных массивов
        max_batch_size = min(cols, int(self.max_gpu_memory * 0.8 / memory_per_col))
        max_batch_size = max(1, max_batch_size // 32 * 32)  # Выравниваем для CUDA
        print(f"Column batch size: {max_batch_size} столбцов из {cols}")

        # Подготавливаем результаты (в исходных координатах)
        all_upper_peaks = []  # Будет содержать точки в формате (x, y)
        all_lower_peaks = []

        # Batch-обработка столбцов (теперь это строки в транспонированной матрице)
        for batch_start in range(0, cols, max_batch_size):
            batch_end = min(batch_start + max_batch_size, cols)
            batch_cols = batch_end - batch_start

            # Выделяем память под batch на GPU (транспонированное представление)
            batch_coefs = coefs_transposed[batch_start:batch_end, :]  # (batch_cols, rows)
            batch_max_mask = max_mask_gpu[batch_start:batch_end, :]
            batch_min_mask = min_mask_gpu[batch_start:batch_end, :]

            # Векторизованная интерполяция для ВСЕХ столбцов в batch одновременно
            batch_upper_env = self._interpolate_cols_batch_gpu(
                batch_coefs, batch_max_mask, rows
            )
            batch_lower_env = self._interpolate_cols_batch_gpu(
                batch_coefs, batch_min_mask, rows
            )

            # Векторизованный поиск пиков для ВСЕХ столбцов в batch
            batch_upper_peaks = self._find_peaks_cols_batch_gpu(batch_upper_env, direction='max')
            batch_lower_peaks = self._find_peaks_cols_batch_gpu(batch_lower_env, direction='min')

            # Преобразуем результаты в исходные координаты (x, y)
            for i in range(batch_cols):
                actual_col = batch_start + i  # Это x-координата в исходной системе
                upper_peaks = batch_upper_peaks[i]  # y-координаты для текущего столбца
                lower_peaks = batch_lower_peaks[i]

                if len(upper_peaks) > 0:
                    # Для столбцов: (actual_col, y)
                    all_upper_peaks.extend([(actual_col, int(y)) for y in upper_peaks])
                if len(lower_peaks) > 0:
                    all_lower_peaks.extend([(actual_col, int(y)) for y in lower_peaks])

            # Очищаем память GPU от временных массивов batch
            del batch_coefs, batch_max_mask, batch_min_mask
            del batch_upper_env, batch_lower_env
            del batch_upper_peaks, batch_lower_peaks
            cp.get_default_memory_pool().free_all_blocks()

        return all_upper_peaks, all_lower_peaks

    @staticmethod
    def _interpolate_cols_batch_gpu(batch_coefs, batch_mask, rows):
        """
        Векторизованная интерполяция для batch столбцов
        batch_coefs: (batch_size, rows) - ТРАНСПОНИРОВАННЫЙ вид
        batch_mask: (batch_size, rows) bool
        Возвращает: (batch_size, rows) интерполированные значения
        """
        batch_size = batch_coefs.shape[0]

        # 1. Создаем равномерную сетку для всех столбцов (теперь это строки)
        t = cp.arange(rows, dtype=cp.float32)

        # One point is constant; two or more define a linear interpolant.
        result = cp.full(batch_coefs.shape, cp.nan, dtype=cp.float32)

        # 3. Для каждого столбца в batch выполняем интерполяцию
        for i in range(batch_size):
            # Получаем маску для текущего столбца
            col_mask = batch_mask[i]

            # Проверяем, достаточно ли точек для интерполяции
            valid_points = cp.sum(col_mask)
            if valid_points >= 1:
                # Получаем координаты и значения
                points_idx = cp.where(col_mask)[0]
                values = batch_coefs[i, points_idx]

                # Сортируем
                sort_idx = cp.argsort(points_idx)
                points_idx_sorted = points_idx[sort_idx]
                values_sorted = values[sort_idx]

                # Линейная интерполяция
                interpolated = cp.interp(t, points_idx_sorted.astype(cp.float32), values_sorted)
                result[i] = interpolated

        return result

    @staticmethod
    def _find_peaks_cols_batch_gpu(batch_data, direction='max'):
        """
        Векторизованный поиск пиков для batch столбцов
        batch_data: (batch_size, rows) - ТРАНСПОНИРОВАННЫЙ вид
        Возвращает: список массивов с индексами пиков для каждого столбца
        """
        batch_size = batch_data.shape[0]
        all_peaks = []

        for i in range(batch_size):
            col_data = batch_data[i]

            if direction == 'max':
                peaks, _ = find_peaks(col_data)
            else:  # 'min'
                peaks, _ = find_peaks(-col_data)

            all_peaks.append(peaks.get() if hasattr(peaks, 'get') else peaks)

        return all_peaks
