from DataCleanser import runDataCleanser
from DataLoader import runDataLoader
import pandas as pd

pd.set_option('display.max_columns', None)
covidDF = runDataLoader()
print("\n---------Abstract after initial load-------------\n")
print(covidDF["abstract"])
covidDF = runDataCleanser(covidDF)
print("\n---------Abstract after Cleansing-------------\n")
print(covidDF["abstract"])
