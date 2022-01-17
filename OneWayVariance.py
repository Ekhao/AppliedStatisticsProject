# This python program does a one way analysis of variance on the dataset to describe whether
# there is a statistical significance between the actual RUL and either of the two predictions

from operator import index
import numpy as np
import pandas as pd

# First the data is loaded
RUL_Pred = pd.read_csv("Data.csv")

# The data in its current form is not appropriate for a one way analysis of variance
# It needs to be transformed to having a categorical explanitory variable
RUL_Pred = pd.DataFrame(
    {
        "Index": range(300),
        "Source": np.repeat(["Actual", "LSTM", "Linear"], 100),
        "RUL": RUL_Pred["Actual RUL"].append(RUL_Pred["LSTM Prediction"]).append(RUL_Pred["Linear Regression Prediction"])
    })
RUL_Pred = RUL_Pred.set_index("Index")

# Then we investigate the data first by looking at it and printing summary information
print(RUL_Pred.head())
print("-"*100)
print(RUL_Pred.dtypes)
print("-"*100)
print(RUL_Pred.describe())

# We then do a few plots of the data
