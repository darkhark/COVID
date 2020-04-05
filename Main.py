from DataCleanser import runDataCleanser
from DataLoader import runDataLoader

covidDF = runDataLoader()
print("\n---------Abstract after initial load-------------\n")
print(covidDF["abstract"])
covidDF = runDataCleanser(covidDF)
print("\n---------Abstract after Cleansing-------------\n")
print(covidDF["abstract"])
