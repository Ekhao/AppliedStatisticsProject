# This python program does a one way analysis of variance on the dataset to describe whether
# there is a statistical significance between the actual RUL and either of the two predictions
# !!Assumtions: There is an equal amount of Actual RULs and predictions for each model

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import researchpy as rp

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

RUL_Pred["log_RUL"] = np.log10(RUL_Pred["RUL"].apply(lambda x: x if x > 1 else 1))

# Then we investigate the data first by looking at it and printing summary information
print(RUL_Pred.head())
print("-"*100)
print(RUL_Pred.dtypes)
print("-"*100)
print(RUL_Pred.describe())
print("-"*100)
print(RUL_Pred["Source"].value_counts())
print("-"*100)
print(rp.summary_cont(RUL_Pred["RUL"].groupby(RUL_Pred["Source"])))


# We then do a few plots of the data, I chose to use the seaborn library for this.
# A boxplot would do nicely
sns.boxplot(data=RUL_Pred, x="Source", y="RUL")
plt.savefig("Figures/OneWayVariance/BoxPlot.png")
plt.close()

# Maybe a stripchart
sns.stripplot(data=RUL_Pred, x="Source", y="RUL")
plt.savefig("Figures/OneWayVariance/StripPlot")
plt.close()

# Histograms for different predictions
# Actual
sns.histplot(RUL_Pred["RUL"][RUL_Pred["Source"]=="Actual"])
plt.savefig("Figures/OnewayVariance/FrequencyPlotActual.png")
plt.close()

# LSTM
sns.histplot(RUL_Pred["RUL"][RUL_Pred["Source"]=="LSTM"])
plt.savefig("Figures/OnewayVariance/FrequencyPlotLSTM.png")
plt.close()

# Linear
sns.histplot(RUL_Pred["RUL"][RUL_Pred["Source"]=="Linear"])
plt.savefig("Figures/OnewayVariance/FrequencyPlotLinear.png")
plt.close()


# The same charts on the log transfored data
sns.boxplot(data=RUL_Pred, x="Source", y="log_RUL")
plt.savefig("Figures/OneWayVariance/LogBoxPlot.png")
plt.close()

sns.stripplot(data=RUL_Pred, x="Source", y="log_RUL")
plt.savefig("Figures/OneWayVariance/LogStripPlot")
plt.close()

# Then we conduct a one way analysis of variance
model2 = ols("RUL ~ C(Source)", data=RUL_Pred).fit()
anova_table = sm.stats.anova_lm(model2, typ=2)
print("-"*100)
print(anova_table)
# It seems that there is a statistical significance between the actual RUL and the predictions
# We must remember to check the model assumptions first though
# The model assumes independence between the observations, normality of the data and homogenity of variance

# It is an independence problem that each engine has a RUL in each group. One tries to predict the other.

# We use a qqplot to check the normality of the data
stats.probplot(model2.resid, plot=plt)
plt.savefig("Figures/OneWayVariance/ResidualPlot")
plt.close()
# TODO: The qq plot looks a bit off, but lets accept it for now

# We use levenes test to check the homogenity of variances
levene = stats.levene(RUL_Pred["RUL"][RUL_Pred["Source"] == "Actual"],
                      RUL_Pred["RUL"][RUL_Pred["Source"] == "LSTM"],
                      RUL_Pred["RUL"][RUL_Pred["Source"] == "Linear"])
print("-"*100)
print(levene)

# The levenes test is statistically significant which indicates that the groups do in fact not have homogeneity of variance
# This is also supported visually by our first boxplot
# Since the assumptions of this model does not hold we cannot interpret the results of the data.

