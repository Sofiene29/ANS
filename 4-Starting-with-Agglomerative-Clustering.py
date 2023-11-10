import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster, metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

# Load data from an ARFF file
path = './artificial/'
name = "long1.arff"
databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# Visualize the initial data
f0 = datanp[:, 0]
f1 = datanp[:, 1]
plt.scatter(f0, f1, s=8)
plt.title("Initial Data: " + str(name))
plt.show()

# Agglomerative Clustering with distance_threshold
tps1 = time.time()
seuil_dist = 10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
k = model.n_clusters_
leaves = model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Agglomerative Clustering (average, distance_threshold= " + str(seuil_dist) + ") " + str(name))
plt.show()
print("Number of clusters =", k, ", Number of leaves =", leaves, " Runtime =", round((tps2 - tps1) * 1000, 2), "ms")

# Agglomerative Clustering with a fixed number of clusters
k = 4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Agglomerative Clustering (average, n_cluster= " + str(k) + ") " + str(name))
plt.show()
print("Number of clusters =", kres, ", Number of leaves =", leaves, " Runtime =", round((tps2 - tps1) * 1000, 2), "ms")

# Silhouette Score Optimization
linkage_method = 'average'
dist_thresholds = np.linspace(1, 50, 50)
best_score = -1
best_threshold = None

for dist_threshold in dist_thresholds:
    model = cluster.AgglomerativeClustering(
        distance_threshold=dist_threshold,
        linkage=linkage_method,
        n_clusters=None
    )
    model.fit(datanp)
    labels = model.labels_

    # Check if there are at least two unique labels
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        score = silhouette_score(datanp, labels)
        if score > best_score:
            best_score = score
            best_threshold = dist_threshold
    else:
        print(f"Skipping iteration for distance threshold {dist_threshold} as only one unique label is present.")

print(f"Best distance threshold for {linkage_method}: {best_threshold}")
print(f"Silhouette Score: {best_score}")

# Regroupement and Separation Scores
model = cluster.AgglomerativeClustering(
    distance_threshold=best_threshold,
    linkage=linkage_method,
    n_clusters=None
)

def regroupement_score(X, labels):
    pairwise_distances_matrix = euclidean_distances(X)
    mask = labels[:, None] == labels
    return pairwise_distances_matrix[mask].mean()

def separation_score(X, labels):
    pairwise_distances_matrix = euclidean_distances(X)
    mask = labels[:, None] != labels
    return pairwise_distances_matrix[mask].mean()

model.fit(datanp)
labels = model.labels_
regroupement = regroupement_score(datanp, labels)
separation = separation_score(datanp, labels)

print(f"Regroupement Score of the obtained solution: {regroupement:.3f}")
print(f"Separation Score of the obtained solution: {separation:.3f}")

# Agglomerative Clustering with different linkage methods
linkages = ["ward", "complete", "average", "single"]
results = []

for linkage_method in linkages:
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(linkage=linkage_method, n_clusters=None, distance_threshold=10)
    model = model.fit(datanp)
    labels = model.labels_
    score = silhouette_score(datanp, labels)
    tps2 = time.time()
    runtime = round((tps2 - tps1) * 1000, 2)
    k = model.n_clusters_
    leaves = model.n_leaves_
    results.append((linkage_method, k, leaves, runtime, score))

# Display the results
print(f"{'Linkage':<10} {'Number of Clusters':<12} {'Number of Leaves':<10} {'Runtime (ms)':<12} {'Silhouette Score':<17}")
print('-' * 65)
for res in results:
    print(f"{res[0]:<10} {res[1]:<12} {res[2]:<10} {res[3]:<12.2f} {res[4]:<17.4f}")
