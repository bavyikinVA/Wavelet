import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import random

def clusterization(data, eps):
  sk_dbscan = DBSCAN(eps, min_samples=1)
  sk_dbscan_pred_res = sk_dbscan.fit_predict(data)
  return sk_dbscan_pred_res, sk_dbscan_pred_res.max() + 1

def find_centroids(data, sk_dbscan_pred_res, clusters_len):
  centroids = []
  for cl in range(clusters_len):
    cluster = []
    for i in range(len(sk_dbscan_pred_res)):
      if sk_dbscan_pred_res[i] == cl:
        cluster.append(data[i])
    sk_kmeans = KMeans(n_clusters=1, n_init='auto', random_state=0)
    sk_kmeans.fit(cluster)
    sk_kmeans_pred_res = sk_kmeans.predict(cluster)
    sk_kmeans_centroids = sk_kmeans.cluster_centers_
    centroids.append(sk_kmeans_centroids[0])
  return np.array(centroids)

def graphic_clusters(path, title, data, sk_dbscan_pred_res, centroids):
  plt.figure(figsize=(7, 8))
  plt.scatter(data[:, 0], data[:, 1], c=sk_dbscan_pred_res, cmap='rainbow')
  plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", color="black", s=100)
  plt.title('DBSCAN')
  plt.gca().invert_yaxis()
  name = path + 'Кластеры_и_центроиды, ' + title + '.png'
  plt.savefig(name)
  plt.close()

def graphic_graph(path, title, coefs, centroids):
    point_max = np.asarray(np.where(coefs == coefs.max())).ravel().tolist()
    points_left = np.copy(centroids).tolist()
    all_points = points_left.copy()
    all_points.append(point_max)
    all_points = np.array(all_points)

    # Отладочные выводы
    print(f"Форма coefs: {coefs.shape}")
    print(f"Форма centroids: {centroids.shape}")
    print(f"Форма all_points: {all_points.shape}")

    plt.figure(figsize=(7, 8))
    plt.gca().invert_yaxis()
    plt.plot(all_points[:, 0], all_points[:, 1], 'ro')

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def sort(point, points):
        l = list(points)
        l.sort(key=lambda coord: distance(point, coord))
        points = np.array(l)
        return points

    points_left = sort(point_max, points_left)
    root = []
    for i in range(3):
        if len(points_left) > 0:
            plt.plot([point_max[0], points_left[0][0]], [point_max[1], points_left[0][1]], 'k-')
            root.append(points_left[0])
            points_left = np.delete(points_left, 0, axis=0)

    while len(points_left) > 0:
        for p in root:
            color = (0.5, (random.randint(0, 100) / 100), 0.5)
            for i in range(2):
                if len(points_left) > 0:
                    points_left = sort(p, points_left)
                    plt.plot([p[0], points_left[0][0]], [p[1], points_left[0][1]], color=color)
                    root = root[1:]
                    root.append(points_left[0])
                    points_left = np.delete(points_left, 0, axis=0)
                else:
                    break
    name = path + 'Сетка, ' + title + '.png'
    plt.savefig(name)
    plt.close()
