import numpy as np
import pandas as pd

# First load the data and convert it into a more appropriate format.
PdM_Predictions = pd.read_csv("Data.csv", header=None)

PdM_Predictions = PdM_Predictions.T
PdM_Predictions.rename(columns={"0": "a"})

# PdM_Predictions
print(PdM_Predictions)
