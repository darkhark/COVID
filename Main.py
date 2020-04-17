from clustering.KMeans import trainTestSplit, findOptimalKUsingElbow, findOptimalKUsingSilhouette
from clustering.feature_selection.NGrams import runNGrams
from clustering.feature_selection.Vectorization import getHashVectorizationMatrix, reduceDimensionalityWithTSNE, \
    reduceDimensionalityWithTF_IDF, reduceDimensionalityWithPCA
from plotters.seabornPlots import plotWithoutClusterSns
from preprocessing.DataCleanser import runDataCleanser
from preprocessing.DataLoader import runFullDataLoader, runQuickLoader, runCleansedDataLoader
from preprocessing.WordCounter import addAbstractAndBodyWordCountColumn


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
    matrix = reduceDimensionalityWithTF_IDF(df)
    return reduceDimensionalityWithPCA(matrix)


# covidDF = loadAndCleanInitialData()
# runQuickLoader(1000)
covidDF = runCleansedDataLoader()
# Equivalent of quick loader for loading the csv
# covidDF = covidDF[:][:1000]
print("\n----------Starting Feature Selection---------\n")

X = getTFidfToTNSEMatrix(covidDF)
plotWithoutClusterSns(X, "plot_pictures/tsnePlot.png")
findOptimalKUsingSilhouette(X)
findOptimalKUsingElbow(X)

X = getTFidfPCAMatrix(covidDF)
plotWithoutClusterSns(X, "plot_pictures/pcaPlot.png")
findOptimalKUsingSilhouette(X)
findOptimalKUsingElbow(X)
