"""
Created on 2023/09/11
@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

# Chargement des données
path = './artificial/'
name = "2d-4c.arff"
databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# Affichage des données initiales en 2D
print("---------------------------------------")
print("Affichage données initiales            " + str(name))
f0 = datanp[:, 0]
f1 = datanp[:, 1]
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : " + str(name))
plt.show()

# Application de KMeans pour une valeur de k fixée
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
k = 4
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

# Affichage des données après clustering
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : " + str(name) + " - Nb clusters =" + str(k))
plt.show()

# Affichage des informations sur le clustering
print("nb clusters =", k, ", nb iter =", iteration, ", inertie = ", inertie, ", runtime = ", round((tps2 - tps1) * 1000, 2), "ms")

# Calcul des distances entre les centroids
dists = metrics.pairwise.euclidean_distances(centroids)
min_distances = dists.min(axis=1)
max_distances = dists.max(axis=1)
mean_distances = dists.mean(axis=1)

# Calcul des distances de chaque point aux centroids
point_to_centroid_dists = metrics.pairwise.euclidean_distances(datanp, centroids)

# Affichage des distances pour chaque cluster
for i in range(k):
    cluster_point_dists = point_to_centroid_dists[labels == i, i]
    print(f"Cluster {i + 1} - Min distance: {cluster_point_dists.min():.2f}, Max distance: {cluster_point_dists.max():.2f}, Mean distance: {cluster_point_dists.mean():.2f}")

# Calcul des scores de séparation entre clusters
centroid_distances = metrics.pairwise.pairwise_distances(centroids)
lower_triangle = np.tril(centroid_distances, -1)
non_zero_values = lower_triangle[lower_triangle > 0]
print(f"Separation between clusters - Min distance: {non_zero_values.min():.2f}, Max distance: {non_zero_values.max():.2f}, Mean distance: {non_zero_values.mean():.2f}")

# Calcul et affichage de l'évolution de l'inertie en fonction du nombre de clusters
inertia_values = []
for k in range(1, 11):
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    inertia_values.append(model.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Evolution de l\'inertie')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()

# Évaluation de la qualité du clustering pour différents nombres de clusters
k_values = range(2, 11)
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

# Calcul des métriques pour chaque nombre de clusters
for k in k_values:
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    labels = model.fit_predict(datanp)
    silhouette_scores.append(metrics.silhouette_score(datanp, labels))
    davies_bouldin_scores.append(metrics.davies_bouldin_score(datanp, labels))
    calinski_harabasz_scores.append(metrics.calinski_harabasz_score(datanp, labels))

# Affichage des métriques
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(k_values, silhouette_scores, marker='o', label='Coefficient de Silhouette', color='red')
plt.legend()
plt.title('Évaluation de la qualité du clustering en fonction de k')
plt.ylabel('Coefficient de Silhouette')

plt.subplot(3, 1, 2)
plt.plot(k_values, davies_bouldin_scores, marker='x', label='Indice de Davies-Bouldin', color='green')
plt.legend()
plt.ylabel('Indice de Davies-Bouldin')

plt.subplot(3, 1, 3)
plt.plot(k_values, calinski_harabasz_scores, marker='s', label='Indice de Calinski-Harabasz')
plt.legend()
plt.xlabel('Nombre de clusters')
plt.ylabel('Indice de Calinski-Harabasz')

plt.tight_layout()
plt.show()

# Application de MiniBatchKMeans pour différentes configurations
from sklearn.cluster import MiniBatchKMeans

batch_sizes = [10, 50, 100, 500]
n_clusters_list = [2, 3, 4, 5]
n_init = 10

for n_clusters in n_clusters_list:
    for batch_size in batch_sizes:
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=n_init, init='k-means++')
        model.fit(datanp)
        labels = model.labels_
        centroids = model.cluster_centers_
        silhouette = metrics.silhouette_score(datanp, labels)
        davies_bouldin = metrics.davies_bouldin_score(datanp, labels)
        calinski_harabasz = metrics.calinski_harabasz_score(datanp, labels)
        print(f"Configuration: n_clusters = {n_clusters}, batch_size = {batch_size}")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies Bouldin Score: {davies_bouldin:.3f}")
        print(f"Calinski Harabasz Score: {calinski_harabasz:.3f}")
        print("-----------------------------------")
        plt.scatter(datanp[:, 0], datanp[:, 1], c=labels, s=8)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
        plt.title(f"Données après clustering avec MiniBatchKMeans (n_clusters={n_clusters}, batch_size={batch_size})")
        plt.show()
