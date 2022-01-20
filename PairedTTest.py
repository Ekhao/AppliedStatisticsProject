import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

RUL_Pred = pd.read_csv("Data.csv")

# We first calculate the result of applying the loss function abs(x-y) - (x-y)/2 to both predictions.
# These values will be what we apply our paired t-test on.

LSTM_Diff = RUL_Pred["Actual RUL"] - RUL_Pred["LSTM Prediction"]
Linear_Diff = RUL_Pred["Actual RUL"] - RUL_Pred["Linear Regression Prediction"]

RUL_Pred["LSTM Loss"] = abs(LSTM_Diff) - (LSTM_Diff)/2
RUL_Pred["Linear Loss"] = abs(Linear_Diff) - (Linear_Diff)/2

#What we are really interested in however is the difference between these two:
RUL_Pred["Loss Difference"] = RUL_Pred["LSTM Loss"] - RUL_Pred["Linear Loss"]

# Make some tables and plots of the data.
print(RUL_Pred[["LSTM Loss", "Linear Loss"]].head())
print("-"*100)
print(RUL_Pred["Loss Difference"].head())
print("-"*100)
print(RUL_Pred.dtypes)
print("-"*100)
print(RUL_Pred["Loss Difference"].describe())
print("-"*100)

sns.boxplot(data=RUL_Pred[["LSTM Loss", "Linear Loss"]])
plt.savefig("Figures/PairedTTest/BoxPlot.png")
plt.close()
# Looks like equal variances, but the means might not be equal.

sns.boxplot(data = RUL_Pred["LSTM Loss"] - RUL_Pred["Linear Loss"])
plt.savefig("Figures/PairedTTest/Boxplot2.png")
plt.close()
# Looks normally distributed now

# TODO: If more time make this plot work
#sns.pointplot(data=RUL_Pred[["LSTM Loss", "Linear Loss"]])
#plt.savefig("Figures/PairedTTest/PointPlot.png")
#plt.close()

# Lets do the paired t-test
model = stats.ttest_rel(RUL_Pred["LSTM Loss"], RUL_Pred["Linear Loss"])
print(model)
print("-"*100)
# Since the pvalue is above 0.05 we cannot reject the null hypothesis that the mean of the losses are the same.
# That is if the model assumptions hold

# The assumptions that must hold are Independence and Normality
# As we only have one value for each engine now we have independence.

# To test for normality we plot the data against the theoretical quantiles of a normal distribution in a qq-plot
sm.qqplot(RUL_Pred["LSTM Loss"] - RUL_Pred["Linear Loss"])
plt.savefig("Figures/PairedTTest/Qqplot.png")
plt.close()
# The qqplot seems a little off with one possible outlier, however the paired t-test is robust against this
# log is not usable as loss function returns negative values, taking the abs value will destroy the normal distribution.
# We cant remove outliers from this dataset as it cant be a "mistake"

# Since the assumptions hold we can accept the null hypothesis that
# there is not a statistically significant difference at the 5% test level.


# To get the confidence interval
import pingouin as pt
print(pt.ttest(RUL_Pred["LSTM Loss"], RUL_Pred["Linear Loss"], paired=True))
print("-"*100)

# Extra:
# We can already get an indication of the variances by looking at our boxplot - here they look fine.
# Alternatively we can use the formal levenes test.
levene = stats.levene(RUL_Pred["LSTM Loss"], RUL_Pred["Linear Loss"])
print(levene)
print("-"*100)
# The levenes test confirms what we saw on the boxplot.