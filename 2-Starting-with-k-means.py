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

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="2d-4c.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
k=4
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)


from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
dists = euclidean_distances(centroids) 
print(dists)
min_distances = dists.min(axis=1)
max_distances = dists.max(axis=1)
mean_distances = dists.mean(axis=1)

from sklearn.metrics.pairwise import euclidean_distances

# Calcul des distances de chaque point aux centroids
point_to_centroid_dists = euclidean_distances(datanp, centroids) 

for i in range(k):
    # Récupération des distances du cluster i
    cluster_point_dists = point_to_centroid_dists[labels == i, i]
    print(f"Cluster {i+1} - Min distance: {cluster_point_dists.min():.2f}, Max distance: {cluster_point_dists.max():.2f}, Mean distance: {cluster_point_dists.mean():.2f}")

# Calcul des scores de séparation
centroid_distances = pairwise_distances(centroids)
lower_triangle = np.tril(centroid_distances, -1)
non_zero_values = lower_triangle[lower_triangle > 0]
print(f"Separation between clusters - Min distance: {non_zero_values.min():.2f}, Max distance: {non_zero_values.max():.2f}, Mean distance: {non_zero_values.mean():.2f}")

inertia_values = []

# AppliCATION k-Means pour des valeurs de k allant de 1 à 10 
for k in range(1, 11):
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    inertia_values.append(model.inertia_)

# Afficher l'évolution de l'inertie
plt.figure()
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Evolution de l\'inertie')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()

#2.3 
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

k_values = range(2, 11)  
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []



for k in k_values:
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    start_time = time.time()
    labels = model.fit_predict(datanp)
    end_time = time.time() 
    print("temps d'execution toal" ,end_time-start_time) 
    # Coefficient de silhouette
    silhouette_scores.append(silhouette_score(datanp, labels))
    
    # Indice de Davies-Bouldin
    davies_bouldin_scores.append(davies_bouldin_score(datanp, labels)) 
    
    # Indice de Calinski-Harabasz
    calinski_harabasz_scores.append(calinski_harabasz_score(datanp, labels)) 
    




plt.figure(figsize=(10, 8))

# Coefficient de Silhouette
plt.subplot(3, 1, 1)
plt.plot(k_values, silhouette_scores, marker='o', label='Coefficient de Silhouette',color='red')
plt.legend()
plt.title('Évaluation de la qualité du clustering en fonction de k')
plt.ylabel('Coefficient de Silhouette')

# Indice de Davies-Bouldin
plt.subplot(3, 1, 2)
plt.plot(k_values, davies_bouldin_scores, marker='x', label='Indice de Davies-Bouldin',color='green')
plt.legend()
plt.ylabel('Indice de Davies-Bouldin')

# Indice de Calinski-Harabasz
plt.subplot(3, 1, 3)
plt.plot(k_values, calinski_harabasz_scores, marker='s', label='Indice de Calinski-Harabasz')
plt.legend()
plt.xlabel('Nombre de clusters')
plt.ylabel('Indice de Calinski-Harabasz')

plt.tight_layout()  # pour ajuster l'espace entre les sous-figures
plt.show()

from sklearn.cluster import MiniBatchKMeans


#2.5
from sklearn.cluster import MiniBatchKMeans, KMeans



# Paramètres pour MiniBatchKMeans
batch_sizes = [10, 50, 100, 500]
n_clusters_list = [2, 3, 4, 5]  # Liste des nombre de clusters à tester
n_init = 10  # Nombre de fois que l'algorithme sera exécuté avec différents centroïdes

# Parcourir les différentes configurations
for n_clusters in n_clusters_list:
    for batch_size in batch_sizes:
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=n_init, init='k-means++') 
        start_time = time.time()
        model.fit(datanp)
        end_time = time.time() 
        print("temps d'execution total :" ,end_time-start_time) 
        

        labels = model.labels_
        centroids = model.cluster_centers_

        # Calculer les métriques
        silhouette = silhouette_score(datanp, labels)
        davies_bouldin = davies_bouldin_score(datanp, labels)
        calinski_harabasz = calinski_harabasz_score(datanp, labels)

        # Afficher les résultats
        print(f"Configuration: n_clusters = {n_clusters}, batch_size = {batch_size}")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies Bouldin Score: {davies_bouldin:.3f}")
        print(f"Calinski Harabasz Score: {calinski_harabasz:.3f}")
        print("-----------------------------------")

        # Affichage du clustering
        plt.scatter(datanp[:, 0], datanp[:, 1], c=labels, s=8)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
        plt.title(f"Données après clustering avec MiniBatchKMeans (n_clusters={n_clusters}, batch_size={batch_size})")
        plt.show()






