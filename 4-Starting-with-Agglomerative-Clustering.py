import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = './artificial/'
name="long1.arff"

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



### FIXER la distance
# 
tps1 = time.time()
seuil_dist=10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


###
# FIXER le nombre de clusters
###
k=4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")



#######################################################################

from sklearn.metrics import silhouette_score

# Choix de la méthode de liaison
linkage_method = 'average'  

# Intervalle de seuils ou de nombres de clusters
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
    score = silhouette_score(datanp, labels)
    if score > best_score:
        best_score = score
        best_threshold = dist_threshold

print(f"Meilleur seuil de distance pour {linkage_method}: {best_threshold}")
print(f"Score de silhouette: {best_score}") 


################################################################### 

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances

# Fonctions pour le calcul de la regroupement et de la séparation 

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

regroupement = regroupement_score(datanp, labels)
separation = separation_score(datanp, labels)

print(f"Score de regroupement de la solution obtenue:{regroupement:.3f}")
print(f"Score de séparation de la solution obtenue: {separation:.3f}") 






####################################################################### 





linkages = ["ward", "complete", "average", "single"]
results = []

# Itération sur les différentes méthodes de linkage
for linkage_method in linkages:
    # Mesure du temps de début
    tps1 = time.time()
    
    # Application du clustering agglomératif
    model = cluster.AgglomerativeClustering(linkage=linkage_method, n_clusters=None, distance_threshold=10)
    model = model.fit(datanp)
    labels = model.labels_
    
    # Calculer le score de silhouette
    score = silhouette_score(datanp, labels)
    
    # Mesure du temps de fin
    tps2 = time.time()
    
    # Calculer le temps d'exécution
    runtime = round((tps2 - tps1) * 1000, 2)  # en millisecondes
    k = model.n_clusters_
    leaves = model.n_leaves_
    results.append((linkage_method, k, leaves, runtime, score))

# Afficher les résultats
print(f"{'Linkage':<10} {'Nb Clusters':<12} {'Nb Leaves':<10} {'Runtime (ms)':<12} {'Silhouette Score':<17}")
print('-' * 65)
for res in results:
    print(f"{res[0]:<10} {res[1]:<12} {res[2]:<10} {res[3]:<12.2f} {res[4]:<17.4f}")

    





