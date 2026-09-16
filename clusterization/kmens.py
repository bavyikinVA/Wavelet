import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

k_maxr_path = r"C:\Users\bavyk\Downloads\knn_results\KNN_Str_Info_Scale_40_Channel_Red_max_by_row.txt"

df = pd.read_csv(k_maxr_path, sep=",", header=None,
                 names=["point_id", "x", "y", "neighbor_id", "distance", "angle_deg"])
df = df[9:].copy().reset_index(drop=True)
df_original = df.copy()  # Сохраняем оригинал с координатами
numeric_columns = ["point_id", "x", "y", "neighbor_id", "distance", "angle_deg"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
df_original = df.copy()
df_cluster = df[["point_id", "distance", "angle_deg"]].copy()

# Вычисляем признаки для кластеризации
features_per_point = df_cluster.groupby("point_id").agg({
    "distance": ["mean", "std"],
    "angle_deg": ["mean", "std"]})

features_per_point.columns = ["dist_mean", "dist_std", "angle_mean", "angle_std"]
features_per_point = features_per_point.fillna(0)

print("Признаки для кластеризации:")
print(features_per_point.head())
print(f"\nКоличество точек: {len(features_per_point)}")


feature_columns = ["dist_mean", "dist_std", "angle_mean", "angle_std"]
X = features_per_point[feature_columns].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertias = []
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(2, 10), inertias, 'bo-')
ax1.set_xlabel('Количество кластеров')
ax1.set_ylabel('Inertia')
ax1.set_title('Метод локтя')

ax2.plot(range(2, 10), silhouette_scores, 'ro-')
ax2.set_xlabel('Количество кластеров')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Коэффициент силуэта')
plt.tight_layout()
plt.show()

# Кластеризация
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

features_per_point["cluster"] = labels

def plot_clusters_2d(X, labels, kmeans, features_names=None):
    """
    Визуализация кластеров в пространстве признаков

    Parameters:
    X: масштабированные признаки (n_samples, 2) - используем первые два признака
    labels: метки кластеров
    kmeans: обученная модель KMeans
    features_names: имена признаков
    """
    plt.figure(figsize=(10, 8))

    # Используем первые два признака для визуализации
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap='jet',
        s=20,
        alpha=1,
        edgecolors='black',
        linewidth=0.5)

    # Добавляем центры кластеров
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1],
                c='red', marker='o', s=5,
                edgecolors='black', linewidth=2,
                label='Центры кластеров')

    if features_names:
        plt.xlabel(features_names[0])
        plt.ylabel(features_names[1])
    else:
        plt.xlabel("Признак 1 (dist_mean)")
        plt.ylabel("Признак 2 (dist_std)")

    plt.title(f"Кластеры в пространстве признаков (n_clusters={len(set(labels))})")
    plt.colorbar(scatter, label='Кластер')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("clusters_feature_space.png")
    return plt.gcf()


def plot_clusters_3d(X, labels, kmeans, features_names=None):
    """
    3D визуализация кластеров в пространстве признаков
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Используем первые три признака для 3D визуализации
    scatter = ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=labels,
        cmap='jet',
        s=5,
        alpha=1,
        edgecolors='black',
        linewidth=0.5)

    # Добавляем центры кластеров
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c='red', marker='X', s=200,
               edgecolors='black', linewidth=2,
               label='Центры кластеров')

    if features_names and len(features_names) >= 3:
        ax.set_xlabel(features_names[0])
        ax.set_ylabel(features_names[1])
        ax.set_zlabel(features_names[2])
    else:
        ax.set_xlabel("dist_mean")
        ax.set_ylabel("dist_std")
        ax.set_zlabel("angle_mean")

    ax.set_title(f"3D визуализация кластеров (n_clusters={len(set(labels))})")
    plt.colorbar(scatter, label='Кластер')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("clusters_3d.png")
    return fig


# Визуализация в 2D (первые два признака)
print("\nВизуализация в 2D пространстве признаков:")
features_names_2d = ["dist_mean (scaled)", "dist_std (scaled)"]
plot_clusters_2d(X_scaled[:, :2], labels, kmeans, features_names_2d)

# Визуализация в 3D (первые три признака)
print("\nВизуализация в 3D пространстве признаков:")
features_names_3d = ["dist_mean (scaled)", "dist_std (scaled)", "angle_mean (scaled)"]
plot_clusters_3d(X_scaled[:, :3], labels, kmeans, features_names_3d)

# Подготовка данных для визуализации на изображении
point_coords = df_original[["point_id", "x", "y"]].drop_duplicates()
result = point_coords.merge(
    features_per_point.reset_index(),
    on="point_id")


# Функция для визуализации кластеров на изображении
def plot_clusters_on_image(points_df, image_path=None, figsize=(12, 10)):
    """
    Визуализация точек на изображении с цветовой кодировкой кластеров

    Parameters:
    points_df: DataFrame с колонками x, y, cluster
    image_path: путь к изображению (опционально)
    figsize: размер фигуры
    """
    plt.figure(figsize=figsize)

    # Если указан путь к изображению, загружаем и отображаем его
    if image_path is not None:
        try:
            from PIL import Image
            img = Image.open(image_path)
            plt.imshow(img, cmap='jet' if img.mode == 'L' else None)
        except Exception as e:
            print(f"Не удалось загрузить изображение: {e}")
            plt.gca().set_facecolor('lightgray')
    else:
        plt.gca().set_facecolor('lightgray')

    # Визуализация точек
    scatter = plt.scatter(
        points_df["x"],
        points_df["y"],
        c=points_df["cluster"],
        cmap='jet',
        s=20,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.3)

    plt.gca().invert_yaxis()
    plt.title(f"Кластеры точек на изображении (n_clusters={n_clusters})")
    plt.xlabel("X координата")
    plt.ylabel("Y координата")
    plt.colorbar(scatter, label='Номер кластера')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig('clusters_on_image.png')

    return plt.gcf()


# Функция для визуализации распределения кластеров по пространству признаков
def plot_cluster_distribution(features_df, labels):
    """
    Визуализация распределения кластеров по каждому признаку
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    feature_names = ['dist_mean', 'dist_std', 'angle_mean', 'angle_std']

    for idx, (ax, feature) in enumerate(zip(axes, feature_names)):
        for cluster in np.unique(labels):
            cluster_data = features_df[features_df['cluster'] == cluster][feature]
            ax.hist(cluster_data, bins=20, alpha=0.5, label=f'Кластер {cluster}')

        ax.set_title(f'Распределение признака: {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Частота')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig("cluster_feature_distribution.png")
    return fig

# Или просто визуализация точек без фонового изображения
plot_clusters_on_image(result, image_path=None)

# Визуализация распределения кластеров
plot_cluster_distribution(features_per_point, labels)
