from clustering.KMeans import trainTestSplit, findOptimalKUsingElbow, findOptimalKUsingSilhouette, getKMeansLabels
from clustering.feature_selection.NGrams import runNGrams
from clustering.feature_selection.Vectorization import getHashVectorizationMatrix, reduceDimensionalityWithTSNE, \
    reduceDimensionalityWithTF_IDF, reduceDimensionalityWithPCA
from metrics.SGDscores import getTrainTestSplitScores
from plotters.matPlotLibPlots import plotWithoutClusterSns, plotWithClusters
from preprocessing.DataCleanser import runDataCleanser, cleanDiseaseList
from preprocessing.DataLoader import runFullDataLoader, runQuickLoader, runCleansedDataLoader, getDiseaseData, \
    getKeywordsList
from preprocessing.WordCounter import addAbstractAndBodyWordCountColumn
import pickle

from topic_modeling.LDAmodeling import getLDAModelsAndKeywords


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


def saveKeywords(keywords, vectorized_data):
    f = open('topics.txt', 'w')

    count = 0

    for i in keywords:

        if vectorized_data[count] is not None:
            f.write(', '.join(i) + "\n")
        else:
            f.write("Not enough instances to be determined. \n")
            f.write(', '.join(i) + "\n")
        count += 1

    f.close()


def countDiseaseWordsInKeywords(diseases, keywords):
    k = 1

    clusterMatches = {}
    for cluster in keywords:
        cluster = [word.strip() for word in cluster]
        wordMatches = []
        total = 0
        for diseaseWord in diseases:
            if diseaseWord in cluster:
                wordMatches.append(diseaseWord)
                total += 1
        wordMatches.append(total)
        clusterMatches[k] = wordMatches
        k += 1
    return clusterMatches


# print("Loading dataframe...")
# covidDF = pickle.load(open("covidDF.p", "rb"))
# Equivalent of quick loader for loading the csv
# covidDF = covidDF[:][:1000]

# print("Loading embedded X...")
# X = pickle.load(open("X.p", "rb"))
# print("Loading reduced X...")
# reducedX = pickle.load(open("xReduced.p", "rb"))
# y_pred = pickle.load(open("y_pred16.p", "rb"))

# covidDF['y'] = y_pred

# vectorizedData, keywords = getLDAModelsAndKeywords(covidDF)
# saveKeywords(keywords, vectorizedData)
# print(keywords[:10])

# getTrainTestSplitScores(reduceDimensionalityWithTF_IDF(covidDF), y_pred)
diseaseList = getDiseaseData()
diseaseList = cleanDiseaseList(diseaseList)
keywords = getKeywordsList()
matches = countDiseaseWordsInKeywords(diseaseList, keywords)
for cell in matches:
    print(cell, ":", matches[cell])
