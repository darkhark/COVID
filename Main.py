from clustering.KMeans import trainTestSplit
from clustering.feature_selection.NGrams import runNGrams
from clustering.feature_selection.Vectorization import getHashVectorizationMatrix, reduceDimensionality
from plotters.seabornPlots import plotWithoutClusterSns
from preprocessing.DataCleanser import runDataCleanser
from preprocessing.DataLoader import runFullDataLoader, runQuickLoader, runCleansedDataLoader
from preprocessing.WordCounter import addAbstractAndBodyWordCountColumn

from preprocessing.DataCleanser import languageDetection


def loadAndCleanInitialData():
    df = runFullDataLoader()
    print("\n---------Body After Initial Load-------------\n")
    print(df["body_text"])
    df = runDataCleanser(df)
    print("\n---------Body After Cleansing-------------\n")
    print(df["body_text"])
    return df


#loadAndCleanInitialData()
#runQuickLoader(1000)
covidDF = runCleansedDataLoader()
#Equivalent of quick loader for loading the csv
#covidDF = covidDF[:][:5000]
covidDF = languageDetection(covidDF)
#print("\n----------Starting Feature Selection---------\n")
#matrix = getHashVectorizationMatrix(runNGrams(covidDF, 2, "body_text"))
#X_train, X_test = trainTestSplit(matrix)

#covidDF = addAbstractAndBodyWordCountColumn(covidDF)
#print("\n---------Counts-------------\n")
#print("Abstract Word Count\n")
#print(covidDF["abstract_word_count"])
#print("\nBody Word Count\n")
#print(covidDF["body_word_count"])

#print("X_train size:", len(X_train[0]))
#print("X_test size:", len(X_test[0]), "\n")

#X_embedded = reduceDimensionality(X_train[0], 25)
#plotWithoutClusterSns(X_embedded)
