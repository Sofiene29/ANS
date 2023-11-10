#################################################
#        Importation des bibliothèques          #
#################################################
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster, metrics, preprocessing
from sklearn.neighbors import NearestNeighbors

#################################################
#          Fonction pour afficher les données   #
#################################################

def plot_initial_data(datanp, title):
    f0 = datanp[:, 0]
    f1 = datanp[:, 1]
    plt.scatter(f0, f1, s=8)
    plt.title(title)
    plt.show()

#################################################
#       Fonction pour effectuer le clustering   #
#################################################

def perform_dbscan_clustering(data, epsilon, min_pts, title):
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    model.fit(data)
    labels = model.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=8)
    plt.title(title)
    plt.show()

    print('Nombre de clusters : %d' % n_clusters)
    print('Nombre de points de bruit : %d' % n_noise)
    print('Epsilon :', epsilon)
    print('MinPts :', min_pts)
    print('---------------------------------------')

#################################################
#          Chargement des données à partir      #
#            du fichier ARFF                     #
#################################################

path = './artificial/'
name1 = "smile1.arff"
name = "2d-4c-no4.arff"
databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

#################################################
#       Affichage des données initiales         #
#################################################

print("---------------------------------------")
print("Données Initiales : " + str(name))
plot_initial_data(datanp, "Données Initiales : " + str(name))

#################################################
#   Clustering DBSCAN sur les données brutes    #
#################################################

print("------------------------------------------------------")
print("DBSCAN (1) sur les Données Brutes... ")
epsilon = 0.09
min_pts = 3
perform_dbscan_clustering(datanp, epsilon, min_pts, "Clustering DBSCAN (1) - Epsilon= " + str(epsilon) + " MinPts= " + str(min_pts))

#################################################
#          Standardisation des données           #
#################################################

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)

#################################################
#       Affichage des données standardisées      #
#################################################

print("Affichage des données standardisées")
plot_initial_data(data_scaled, "Données Standardisées")

#################################################
# Clustering DBSCAN sur les données standardisées#
#################################################

print("------------------------------------------------------")
print("DBSCAN (2) sur les Données Standardisées... ")
epsilon = 0.05
min_pts = 5
perform_dbscan_clustering(data_scaled, epsilon, min_pts, "Clustering DBSCAN (2) - Epsilon= " + str(epsilon) + " MinPts= " + str(min_pts))

#################################################
#  Boucle sur différentes valeurs d'epsilon et   #
#               de min_pts                       #
#################################################

print("------------------------------------------------------")
print("Boucle sur différentes valeurs d'epsilon et de min_pts...")
epsilon_values = [1, 2, 20]
min_pts_values = [5, 10, 15]

#for epsilon in epsilon_values:
  #  for min_pts in min_pts_values:
       # perform_dbscan_clustering(datanp, epsilon, min_pts, "Clustering DBSCAN (1) - Epsilon= " + str(epsilon) + " MinPts= " + str(min_pts))

#################################################
# Calcul des distances moyennes aux k plus       #
# proches voisins pour chaque exemple            #
#################################################
# Function for the elbow method to determine epsilon
def elbow_method(distances, k):
    distancetrie = np.sort(distances)
    plt.plot(distancetrie)
    plt.title("Elbow Method for Epsilon (k=" + str(k) + ")")
    plt.xlabel("Data Point Index")
    plt.ylabel("Average Distance to k-Nearest Neighbors")
    plt.show()

# Distances aux k plus proches voisins
k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(datanp)
distances, indices = neigh.kneighbors(datanp)

# Calculate average distance to k-nearest neighbors
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(distances.shape[0])])

# Use the elbow method to determine a suitable value for epsilon
elbow_method(newDistances, k)
########################################################

# Function to perform DBSCAN clustering
def perform_dbscan_clustering(data, epsilon, min_pts, title):
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    model.fit(data)
    labels = model.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=8)
    plt.title(title)
    plt.show()

    print('Nombre de clusters : %d' % n_clusters)
    print('Nombre de points de bruit : %d' % n_noise)
    print('Epsilon :', epsilon)
    print('MinPts :', min_pts)
    print('---------------------------------------')

# Perform DBSCAN clustering with epsilon = 1.80 and min_pts = 10
epsilon = 2
min_pts = 9
perform_dbscan_clustering(datanp, epsilon, min_pts, "Clustering DBSCAN - Epsilon= " + str(epsilon) + " MinPts= " + str(min_pts))


########################################################
################################################
#       Fonction pour effectuer le clustering   #
#              avec HDBSCAN                      #
#################################################

def perform_hdbscan_clustering(data, min_cluster_size, title):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    model.fit(data)
    labels = model.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=8)
    plt.title(title)
    plt.show()

    print('Nombre de clusters : %d' % n_clusters)
    print('Nombre de points de bruit : %d' % n_noise)
    print('Min Cluster Size :', min_cluster_size)
    print('---------------------------------------')

#################################################
# Clustering HDBSCAN sur les données standardisées#
#################################################

print("------------------------------------------------------")
print("HDBSCAN sur les Données Standardisées... ")
min_cluster_size_hdbscan = 5
perform_hdbscan_clustering(data_scaled, min_cluster_size_hdbscan, "Clustering HDBSCAN - Min Cluster Size= " + str(min_cluster_size_hdbscan))

#################################################
# Comparaison entre DBSCAN et HDBSCAN            #
#################################################

print("------------------------------------------------------")
print("Comparaison entre DBSCAN et HDBSCAN... ")

# Paramètres pour DBSCAN
epsilon_dbscan = 0.5
min_pts_dbscan = 5

# Paramètres pour HDBSCAN
min_cluster_size_hdbscan = 9

# Clustering avec DBSCAN
perform_dbscan_clustering(data_scaled, epsilon_dbscan, min_pts_dbscan, "Clustering DBSCAN - Epsilon= " + str(epsilon_dbscan) + " MinPts= " + str(min_pts_dbscan))

# Clustering avec HDBSCAN
perform_hdbscan_clustering(data_scaled, min_cluster_size_hdbscan, "Clustering HDBSCAN - Min Cluster Size= " + str(min_cluster_size_hdbscan))