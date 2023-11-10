import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

def load_data(file_path):
    databrut = arff.loadarff(open(file_path, 'r'))
    return np.array([[x[0], x[1]] for x in databrut[0]])

def plot_scatter(data, title):
    f0, f1 = data[:, 0], data[:, 1]
    plt.scatter(f0, f1, s=8)
    plt.title(title)
    plt.show()

def plot_cluster(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=8)
    plt.title(title)
    plt.show()

def agglomerative_clustering(data, linkage, n_clusters=None, distance_threshold=None):
    model = cluster.AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, distance_threshold=distance_threshold)
    model.fit(data)
    labels = model.labels_
    return labels

def silhouette_analysis(data, max_clusters):
    temps = []
    scores = []

    for d in range(2, max_clusters + 1):
        tps1 = time.time()
        labels = agglomerative_clustering(data, 'single', n_clusters=d)
        tps2 = time.time()
        temps.append(round((tps2 - tps1) * 1000, 2))
        silhouette_score = metrics.silhouette_score(data, labels)
        scores.append(silhouette_score)

        # Imprimer les résultats à la console
        print(f"Clusters: {d}, Temps: {round((tps2 - tps1) * 1000, 2)} ms, Score de Silhouette: {silhouette_score}")

    plt.title("Évolution du score de silhouette")
    plt.xlabel("nombre de clusters")
    plt.ylabel("Score de silhouette")
    plt.plot(range(2, max_clusters + 1), scores, 'g-o')
    plt.show()

# Exemple d'utilisation
path = './artificial/'
name = "target.arff"
data = load_data(path + str(name))

# Afficher les données initiales
plot_scatter(data, "Données initiales : " + str(name))

# Appliquer l'analyse de silhouette et imprimer les résultats à la console
silhouette_analysis(data, max_clusters=12)

# Exemple d'utilisation avec la meilleure valeur de silhouette (2 clusters)
best_num_clusters = 4
best_labels = agglomerative_clustering(data, 'single', n_clusters=best_num_clusters)

# Afficher le résultat du clustering
plot_cluster(data, best_labels, f"Clustering Agglomératif avec {best_num_clusters} clusters")

def trouver_seuil_optimal(data, linkage, max_seuil=2.0, pas=0.1, metrique='silhouette'):
    seuils = np.arange(0.1, max_seuil, pas)
    meilleur_score = -1
    meilleur_seuil = 0

    for seuil in seuils:
        etiquettes = agglomerative_clustering(data, linkage, distance_threshold=seuil)

        if metrique == 'silhouette':
            score = metrics.silhouette_score(data, etiquettes)
        # Ajoutez d'autres métriques si nécessaire

        if score > meilleur_score:
            meilleur_score = score
            meilleur_seuil = seuil

    print(f"Meilleur score {metrique} : {meilleur_score} au seuil : {meilleur_seuil}")

    # Vous pouvez retourner le meilleur seuil si nécessaire
    return meilleur_seuil

# Exemple d'utilisation
meilleur_seuil = trouver_seuil_optimal(data, linkage='single', max_seuil=2.0, pas=0.1, metrique='silhouette')
print(f"Meilleur seuil : {meilleur_seuil}")

##########################################################################
# Fonction pour mesurer le temps de calcul du clustering agglomératif
# pour différentes méthodes de linkage et un nombre variable de clusters.
#
# Paramètres :
# - data : Les données à utiliser pour le clustering.
# - max_clusters : Le nombre maximum de clusters à considérer.
#
# Méthode :
# La fonction itère sur différentes méthodes de linkage ('single', 'complete', 'average', 'ward')
# et mesure le temps de calcul du clustering agglomératif pour chaque nombre de clusters de 2 à max_clusters.
# Les résultats sont affichés à la console et tracés dans une figure où l'axe x représente le nombre de clusters
# et l'axe y représente le temps de calcul en millisecondes.
#
# Exemple d'utilisation :
# mesure_temps_calcul(data, max_clusters=12)
##########################################################################

def mesure_temps_calcul(data, max_clusters):
    # Méthodes de linkage à considérer
    methodes_linkage = ['single', 'complete', 'average', 'ward']
    
    for linkage in methodes_linkage:
        temps = []

        # Itération sur différents nombres de clusters
        for d in range(2, max_clusters + 1):
            tps1 = time.time()
            labels = agglomerative_clustering(data, linkage, n_clusters=d)
            tps2 = time.time()
            temps.append((tps2 - tps1) * 1000)  # temps en millisecondes

            # Imprimer les résultats à la console
            print(f"Linkage: {linkage}, Clusters: {d}, Temps: {round((tps2 - tps1) * 1000, 2)} ms")

        # Tracer la figure pour la méthode de linkage actuelle
        plt.title(f"Temps de calcul pour la méthode de linkage '{linkage}'")
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Temps de calcul (ms)")
        plt.plot(range(2, max_clusters + 1), temps, 'g-o')
        plt.show()

# Exemple d'utilisation
mesure_temps_calcul(data, max_clusters=20)

# Appliquer l'analyse de silhouette avec la meilleure valeur
best_labels = agglomerative_clustering(data, 'single', n_clusters=best_num_clusters)
silhouette = metrics.silhouette_score(data, best_labels)
davies_bouldin = metrics.davies_bouldin_score(data, best_labels)

# Imprimer les scores à la console
print(f"Silhouette Score : {silhouette}")
print(f"Davies-Bouldin Score : {davies_bouldin}")

# Afficher le résultat du clustering
plot_cluster(data, best_labels, f"Clustering Agglomératif avec {best_num_clusters} clusters")

