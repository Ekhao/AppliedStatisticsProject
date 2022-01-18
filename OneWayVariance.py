# This python program does a one way analysis of variance on the dataset to describe whether
# there is a statistical significance between the actual RUL and either of the two predictions
# !!Assumtions: There is an equal amount of Actual RULs and predictions for each model

from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# First the data is loaded
RUL_Pred = pd.read_csv("Data.csv")

# The data in its current form is not appropriate for a one way analysis of variance
# It needs to be transformed to having a categorical explanitory variable
# TODO: Make this dataframe be able to take different amounts of actual RULs and predictions.
RUL_Pred = pd.DataFrame(
    {
        "Index": range(300),
        "Source": np.repeat(["Actual", "LSTM", "Linear"], len(RUL_Pred["Actual RUL"])),
        "RUL": RUL_Pred["Actual RUL"].append(RUL_Pred["LSTM Prediction"]).append(RUL_Pred["Linear Regression Prediction"])
    })
RUL_Pred = RUL_Pred.set_index("Index")
RUL_Pred["Source"] = RUL_Pred["Source"].astype("category")

# Then we investigate the data first by looking at it and printing summary information
print(RUL_Pred.head())
print("-"*100)
print(RUL_Pred.dtypes)
print("-"*100)
print(RUL_Pred.describe())
print("-"*100)
print(RUL_Pred["Source"].value_counts())


# We then do a few plots of the data
# A boxplot would do nicely
boxplot = RUL_Pred.boxplot(
    by="Source", patch_artist=True, grid=False)
boxplot.set_ylabel("Remaining Useful Life (RUL)")
boxplot.set_xlabel("Model")
boxplot.set_title("Remaining Useful Life ground truth and predictions")
plt.savefig("Figures/OneWayBoxPlot.png")
# Maybe a stripchart


# Then we conduct a one way analysis of variance
