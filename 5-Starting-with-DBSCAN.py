import numpy as np
import matplotlib.pyplot as plt
import time
import hdbscan
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering


path = './artificial/'
name="twodiamonds.arff"

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


# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
# 
print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")
tps1 = time.time()
epsilon= 0.4#2  # 4
min_pts= 10 #10   # 10
clusterer = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=100)
labels = clusterer.fit_predict(datanp)

# Visualisation des résultats
plt.scatter(datanp[:,0], datanp[:,1], c=labels, s=8)
plt.title("Résultat HDBSCAN")
plt.show()
#model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering DBSCAN (1) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()


####################################################
# Standardisation des donnees

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()


print("------------------------------------------------------")
print("Appel DBSCAN (2) sur données standardisees ... ")
tps1 = time.time()
epsilon=0.05 #0.05
min_pts=5 # 10
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(data_scaled)

tps2 = time.time()
labels = model.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()

################################################################

# Valeurs pour tester
#eps_values = [0.01,0.05,0.1, 0.5, 1, 1.1]
#min_samples_values = [5, 10, 15, 20]


#scaler = preprocessing.StandardScaler().fit(datanp)
#data_scaled = scaler.transform(datanp)
#for epsilon in eps_values:
 #   for min_pts in min_samples_values:
  #      model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
   #     model.fit(data_scaled)
    #    labels = model.labels_
        # Number of clusters in labels, ignoring noise if present.
     #   n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
      #  n_noise = list(labels).count(-1) 
       # print(' pour eps:%.2f ' % epsilon )
        #print(' pour min_samp:%d ' % min_pts )
        #print('Number of clusters: %d ' % n_clusters)
        #print('Number of noise points: %d' % n_noise)   

#########################################################################


# Estimation de eps
k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(datanp)
distances, _ = neigh.kneighbors(datanp)
distances = np.sort(distances, axis=0)[:,1]
plt.title("Plus proches voisins " + str(k))
plt.plot(distances)
plt.show()

# Suggérer une valeur pour eps en observant la courbe
eps_suggested = 2

# DBSCAN avec les données brutes 

#min_samples_values = [1 , 3 , 5, 10, 15, 20] 
#for k in min_samples_values:
model = cluster.DBSCAN(eps=eps_suggested, min_samples=k)
labels = model.fit_predict(datanp)
plt.scatter(f0, f1, c=labels, s=8)
plt.title(f"DBSCAN (raw data) with eps={eps_suggested} and min_samples={k}")
plt.show()

# Standardisation des données
'''scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
f0_scaled = data_scaled[:,0]
f1_scaled = data_scaled[:,1]

# DBSCAN avec les données standardisées
model_scaled = cluster.DBSCAN(eps=eps_suggested, min_samples=k)
labels_scaled = model_scaled.fit_predict(data_scaled)
plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
plt.title(f"DBSCAN (standardized data) with eps={eps_suggested} and min_samples={k}")
plt.show()'''








