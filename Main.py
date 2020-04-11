from clustering.feature_selection.NGrams import runNGrams
from preprocessing.DataCleanser import runDataCleanser
from preprocessing.DataLoader import runDataLoader, runQuickLoader
from preprocessing.WordCounter import addAbstractAndBodyWordCountColumn

# covidDF = runDataLoader()
covidDF = runQuickLoader()
print("\n---------Body After Initial Load-------------\n")
print(covidDF["body_text"])
covidDF = runDataCleanser(covidDF)
print("\n---------Abstract After Cleansing-------------\n")
print(covidDF["body_text"])
covidDF = addAbstractAndBodyWordCountColumn(covidDF)
print("\n---------Counts-------------\n")
print("Abstract Word Count\n")
print(covidDF["abstract_word_count"])
print("\nBody Word Count\n")
print(covidDF["body_word_count"])

print("\n----------Starting Feature Selection---------\n")
runNGrams(covidDF, 2)
