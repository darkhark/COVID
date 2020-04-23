from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from tqdm import tqdm
from plotters.matPlotLibPlots import plotElbowForKmeans, plotSilhouetteScores


def trainTestSplit(matrix, method="single", splits=3):
    X_Trains = []
    X_Tests = []
    if method == "single":
        X_train, X_test = train_test_split(matrix.toarray(), test_size=0.2, random_state=42)
        X_Trains.append(X_train)
        X_Tests.append(X_test)
    elif method == "kfolds":
        kf = KFold(splits, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(matrix):
            X_Trains.append(matrix[train_index])
            X_Tests.append(matrix[test_index])
    return X_Trains, X_Tests


def findOptimalKUsingElbow(X, name):
    distortions = []
    kMax = 50
    kRange = range(2, kMax + 1)
    print("Plotting distortions for optimal K...")
    for k in tqdm(kRange):
        k_means = KMeans(n_clusters=k, random_state=42).fit(X)
        k_means.fit(X)
        distortions.append(sum(np.min(cdist(X, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    plotElbowForKmeans(kRange, distortions, name)


def findOptimalKUsingSilhouette(X, name):
    sil = []
    kMax = 50
    kRange = range(2, kMax + 1)
    print("Plotting silhouette scores for optimal K...")
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in tqdm(kRange):
        kmeans = KMeans(n_clusters=k).fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric='euclidean'))
    plotSilhouetteScores(kRange, sil, name)


def getKMeansLabels(reducedX, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(reducedX)
    return y_pred
