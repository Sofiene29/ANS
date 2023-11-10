import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram

# Exemple: Dendrogramme and Agglomerative Clustering

path = './artificial/'
name = "xclara.arff"

databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
print("---------------------------------------")
print("Affichage données initiales            " + str(name))
f0 = datanp[:, 0]
f1 = datanp[:, 1]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : " + str(name))
plt.show()

# Dendrogram function
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix)

# Hierarchical Clustering Dendrogram
model = cluster.AgglomerativeClustering(distance_threshold=0, linkage='average', n_clusters=None)
model = model.fit(datanp)

plt.figure(figsize=(12, 12))
plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(model)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# FIXER la distance
tps1 = time.time()
seuil_dist = 10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
k = model.n_clusters_
leaves = model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= " + str(seuil_dist) + ") " + str(name))
plt.show()
print("nb clusters =", k, ", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1) * 1000, 2), "ms")

# FIXER le nombre de clusters
k = 3
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= " + str(k) + ") " + str(name))
plt.show()
print("nb clusters =", kres, ", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1) * 1000, 2), "ms")

# Loop over datasets and linkages
datasets = ["square1.arff", "xclara.arff", "diamond9.arff", "dartboard1.arff"]
linkages = ["ward", "average", "complete", "single"]

for data in datasets:
    databrut = arff.loadarff(open(path + data, 'r'))
    datanp = np.array([[x[0], x[1]] for x in databrut[0]])
    f0 = datanp[:, 0]
    f1 = datanp[:, 1]

    for link in linkages:
        model = cluster.AgglomerativeClustering(linkage=link, n_clusters=None, distance_threshold=10)
        model = model.fit(datanp)
        plt.title(f"dondogramme({link}, data={data})")
        plot_dendrogram(model)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.scatter(f0, f1, c=model.labels_, s=8)
        plt.title(f"Clustering agglomératif ({link}, data={data})")
        plt.show()
