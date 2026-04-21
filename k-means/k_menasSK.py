import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv("./datasets/iris.csv")
    X = df.drop(columns=['Id', 'Species'])
    y_real = df['Species']
except:
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    y_real = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

start_time_sk = time.time()
kmeans_sk = KMeans(n_clusters=3, n_init=10, random_state=42)
labels_sk = kmeans_sk.fit_predict(X_scaled)
end_time_sk = time.time()

score_sk = silhouette_score(X_scaled, labels_sk)
print(f"Sklearn KMeans (K=3) - Silhouette Score: {score_sk:.4f}")
print(f"Tempo de execução Sklearn: {end_time_sk - start_time_sk:.4f}s\n")

for n_comp in [1, 2]:
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    km_pca = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels_pca = km_pca.fit_predict(X_pca)
    centroids_pca = km_pca.cluster_centers_
    
    plt.figure(figsize=(8, 5))
    if n_comp == 1:
        plt.scatter(X_pca, np.zeros_like(X_pca), c=labels_pca, cmap='viridis', alpha=0.6)
        plt.scatter(centroids_pca, [0,0,0], c='red', marker='X', s=200, label='Centróides')
        plt.title("PCA com 1 Componente")
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca, cmap='viridis', alpha=0.6)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centróides')
        plt.title("PCA com 2 Componentes")
    
    plt.legend()
    plt.show()