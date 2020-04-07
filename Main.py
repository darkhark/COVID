from DataCleanser import runDataCleanser
from DataLoader import runDataLoader
from WordCounter import addAbstractAndBodyWordCountColumn

covidDF = runDataLoader()
print("\n---------Abstract after initial load-------------\n")
print(covidDF["abstract"])
covidDF = runDataCleanser(covidDF)
print("\n---------Abstract after Cleansing-------------\n")
print(covidDF["abstract"])
covidDF = addAbstractAndBodyWordCountColumn(covidDF)
print("\n---------Counts-------------\n")
print("Abstract Word Count\n")
print(covidDF["abstract_word_count"])
print("\nBody Word Count\n")
print(covidDF["body_word_count"])
