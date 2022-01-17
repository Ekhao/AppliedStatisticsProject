# This python program was used to convert the original dataset into a properly formatted dataset with categories as columns and observations as rows

import numpy as np
import pandas as pd

PdM_Predictions = pd.read_csv("Data.csv", header=None)

PdM_Predictions = PdM_Predictions.T
PdM_Predictions = PdM_Predictions.rename(
    columns={0: "Actual RUL", 1: "LSTM Prediction", 2: "Linear Regression Prediction"})

PdM_Predictions.to_csv("Data.csv", index=False)
