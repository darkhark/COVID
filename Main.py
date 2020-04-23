from clustering.KMeans import trainTestSplit, findOptimalKUsingElbow, findOptimalKUsingSilhouette, getKMeansLabels
from clustering.feature_selection.NGrams import runNGrams
from clustering.feature_selection.Vectorization import getHashVectorizationMatrix, reduceDimensionalityWithTSNE, \
    reduceDimensionalityWithTF_IDF, reduceDimensionalityWithPCA
from plotters.matPlotLibPlots import plotWithoutClusterSns, plotWithClusters
from preprocessing.DataCleanser import runDataCleanser
from preprocessing.DataLoader import runFullDataLoader, runQuickLoader, runCleansedDataLoader
from preprocessing.WordCounter import addAbstractAndBodyWordCountColumn
import pickle


def loadAndCleanInitialData():
    df = runFullDataLoader()
    print("\n---------Body After Initial Load-------------\n")
    print(df["body_text"])
    df = runDataCleanser(df)
    print("\n---------Body After Cleansing-------------\n")
    print(df["body_text"])
    return df


def appendBodyAndAbstractWordCounts(df):
    df = addAbstractAndBodyWordCountColumn(df)
    print("\n---------Counts-------------\n")
    print("Abstract Word Count\n")
    print(df["abstract_word_count"])
    print("\nBody Word Count\n")
    print(df["body_word_count"])


def getTrainTestSplitWithTSNE(df):
    matrix = getHashVectorizationMatrix(runNGrams(df, 2, "body_text"))
    X_train, X_test = trainTestSplit(matrix)
    print("X_train size:", len(X_train[0]))
    print("X_test size:", len(X_test[0]), "\n")
    X_embedded = reduceDimensionalityWithTSNE(X_train[0], 15)
    plotWithoutClusterSns(X_embedded, "plot_pictures/ngramsPlot.png")
    return X_train, X_test


def getTFidfToTNSEMatrix(df):
    matrix = reduceDimensionalityWithTF_IDF(df)
    return reduceDimensionalityWithTSNE(matrix, 3)


def getTFidfPCAMatrix(df):
    matrix = reduceDimensionalityWithTF_IDF(df, calculateMaxFeatures=False)
    return reduceDimensionalityWithPCA(matrix)


def saveEmbeddedXAndPlots():
    X = getTFidfToTNSEMatrix(covidDF)
    pickle.dump(X, open("X.p", "wb"))
    plotWithoutClusterSns(X, "plot_pictures/tsnePlot.png", "t-SNE Covid-19 Articles")
    findOptimalKUsingSilhouette(X, "TSNE")
    findOptimalKUsingElbow(X, "TSNE")
    return X


def saveXReduced():
    x_reduced = getTFidfPCAMatrix(covidDF)
    pickle.dump(x_reduced, open("xReduced.p", "wb"))
    # plotWithoutClusterSns(reducedX, "plot_pictures/pcaPlot.png", "PCA Covid-19 Articles")
    # findOptimalKUsingSilhouette(reducedX, "PCA")
    # findOptimalKUsingElbow(reducedX, "PCA")
    return x_reduced


def savePredictedY(X_embedded, X_reduced, k):
    predictedY = getKMeansLabels(X_reduced, k)
    yPredName = "y_pred" + str(k) + ".p"
    pickle.dump(predictedY, open(yPredName, "wb"))
    plotWithClusters(X_embedded, predictedY, k)
    return predictedY


print("Loading dataframe...")
covidDF = pickle.load(open("covidDF.p", "rb"))
# Equivalent of quick loader for loading the csv
# covidDF = covidDF[:][:1000]

print("Loading embedded X...")
X = pickle.load(open("X.p", "rb"))
print("Loading reduced X...")
reducedX = pickle.load(open("xReduced.p", "rb"))
for k in range(5, 11):
    savePredictedY(X, reducedX, k)

for k in range(15, 26):
    savePredictedY(X, reducedX, k)

# covidDF['y'] = y_pred
