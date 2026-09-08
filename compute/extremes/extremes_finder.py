class ExtremesFinder:
    @staticmethod
    def find_extremes_gpu(coefs_gpu, row_var, col_var, max_var, min_var):
        """
        GPU версия поиска экстремумов
        """
        import cupy as cp

        if not isinstance(coefs_gpu, cp.ndarray):
            coefs_gpu = cp.asarray(coefs_gpu, dtype=cp.float32)

        points_max_by_row = []
        points_min_by_row = []
        points_max_by_column = []
        points_min_by_column = []

        # Экстремумы построчно
        if row_var and (max_var or min_var):
            left = coefs_gpu[:, :-2]
            center = coefs_gpu[:, 1:-1]
            right = coefs_gpu[:, 2:]

            if max_var:
                max_mask = (center > left) & (center > right)
                if cp.any(max_mask):
                    max_coords = cp.where(max_mask)
                    max_y = max_coords[0].get()
                    max_x = max_coords[1].get()
                    points_max_by_row = [[int(x + 1), int(y)] for y, x in zip(max_y, max_x)]

            if min_var:
                min_mask = (center < left) & (center < right)
                if cp.any(min_mask):
                    min_coords = cp.where(min_mask)
                    min_y = min_coords[0].get()
                    min_x = min_coords[1].get()
                    points_min_by_row = [[int(x + 1), int(y)] for y, x in zip(min_y, min_x)]

        # Экстремумы по столбцам
        if col_var and (max_var or min_var):
            up = coefs_gpu[:-2, :]
            center = coefs_gpu[1:-1, :]
            down = coefs_gpu[2:, :]

            if max_var:
                max_mask = (center > up) & (center > down)
                if cp.any(max_mask):
                    max_coords = cp.where(max_mask)
                    max_y = max_coords[0].get()
                    max_x = max_coords[1].get()
                    points_max_by_column = [[int(x), int(y + 1)] for y, x in zip(max_y, max_x)]

            if min_var:
                min_mask = (center < up) & (center < down)
                if cp.any(min_mask):
                    min_coords = cp.where(min_mask)
                    min_y = min_coords[0].get()
                    min_x = min_coords[1].get()
                    points_min_by_column = [[int(x), int(y + 1)] for y, x in zip(min_y, min_x)]

        return (coefs_gpu.get() if isinstance(coefs_gpu, cp.ndarray) else coefs_gpu,
                points_max_by_row, points_max_by_column,
                points_min_by_row, points_min_by_column)
