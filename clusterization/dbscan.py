import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class DBSCAN:
    def __init__(self, epsilon=1.0, min_points=5):
        self.epsilon = epsilon
        self.min_points = min_points
        self.labels = None
        self.data = None

    def fit(self, data):
        self.data = np.asarray(data)
        n_points = len(self.data)

        # -1 = шум, 0,1,2,... = кластеры
        self.labels = np.full(n_points, -1, dtype=int)

        # visited[i] = была ли точка уже обработана
        visited = np.zeros(n_points, dtype=bool)

        cluster_id = 0

        for point_idx in range(n_points):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            neighbors = self._range_query(point_idx)

            # Недостаточно соседей -> шум
            if len(neighbors) < self.min_points:
                self.labels[point_idx] = -1
            else:
                self._expand_cluster(point_idx, neighbors, cluster_id, visited)
                cluster_id += 1

        return self

    def _expand_cluster(self, point_idx, neighbors, cluster_id, visited):
        self.labels[point_idx] = cluster_id

        # Очередь для обхода плотной области
        queue = deque(neighbors)

        while queue:
            current_point = queue.popleft()

            if not visited[current_point]:
                visited[current_point] = True
                current_neighbors = self._range_query(current_point)

                # Если текущая точка ядровая, расширяем кластер дальше
                if len(current_neighbors) >= self.min_points:
                    queue.extend(current_neighbors)

            # Если точка ещё не отнесена ни к одному кластеру, присваиваем ей текущий кластер
            if self.labels[current_point] == -1:
                self.labels[current_point] = cluster_id

    def _range_query(self, point_idx):
        distances = np.linalg.norm(self.data - self.data[point_idx], axis=1)
        neighbors = np.where(distances <= self.epsilon)[0]
        return neighbors.tolist()

file = open(r"C:\Users\bavyk\Downloads\ВП_20_03_2026_00_18\Задача 1 20_03_2026_00_19\Scale_40\Str_Точки_максимума_по_cтолбцам_масштаб_40_Красный.txt").readlines()
coords = []

for pairs in file:
    coords.append([int(x) for x in pairs.split(',')])

print("Первые 2 точки:", coords[:2])

coords = np.array(coords)
x = coords[:, 0]
y = coords[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=1)
plt.title("Исходное распределение точек")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.savefig("source.png")
plt.close()

dbscan = DBSCAN(epsilon=13, min_points=5)
dbscan.fit(coords)

print("Метки кластеров:")
print(dbscan.labels)

unique_labels = set(dbscan.labels)

plt.figure(figsize=(8, 6))

for label in unique_labels:
    mask = dbscan.labels == label
    points = coords[mask]

    if label == -1:
        plt.scatter(
            points[:, 0], points[:, 1],
            s=2, marker='x', label='Шум'
        )
    else:
        plt.scatter(
            points[:, 0], points[:, 1],
            s=2, label=f'Кластер {label}'
        )

plt.title("Результат кластеризации DBSCAN")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.savefig("dbscan.png")
plt.close()
