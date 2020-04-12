from clustering.KMeans import trainTestSplit
from clustering.feature_selection.NGrams import runNGrams
from clustering.feature_selection.Vectorization import getHashVectorizationMatrix, reduceDimensionality
from plotters.seabornPlots import plotWithoutClusterSns
from preprocessing.DataCleanser import runDataCleanser
from preprocessing.DataLoader import runDataLoader, runQuickLoader
from preprocessing.WordCounter import addAbstractAndBodyWordCountColumn


#covidDF = runDataLoader()
covidDF = runQuickLoader()
print("\n---------Body After Initial Load-------------\n")
print(covidDF["body_text"])
covidDF = runDataCleanser(covidDF)
print("\n---------Abstract After Cleansing-------------\n")
print(covidDF["body_text"])

print("\n----------Starting Feature Selection---------\n")
matrix = getHashVectorizationMatrix(runNGrams(covidDF, 2))
X_train, X_test = trainTestSplit(matrix)

covidDF = addAbstractAndBodyWordCountColumn(covidDF)
print("\n---------Counts-------------\n")
print("Abstract Word Count\n")
print(covidDF["abstract_word_count"])
print("\nBody Word Count\n")
print(covidDF["body_word_count"])

print("X_train size:", len(X_train[0]))
print("X_test size:", len(X_test[0]), "\n")

X_embedded = reduceDimensionality(X_train[0])
plotWithoutClusterSns(X_embedded)
