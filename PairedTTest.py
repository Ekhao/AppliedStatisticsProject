import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

RUL_Pred = pd.read_csv("Data.csv")

# We first calculate the result of applying the loss function abs(x-y) - (x-y)/2 to both predictions.
# These values will be what we apply our paired t-test on.

LSTM_Diff = RUL_Pred["Actual RUL"] - RUL_Pred["LSTM Prediction"]
Linear_Diff = RUL_Pred["Actual RUL"] - RUL_Pred["Linear Regression Prediction"]

RUL_Pred["LSTM Loss"] = abs(LSTM_Diff) - (LSTM_Diff)/2
RUL_Pred["Linear Loss"] = abs(Linear_Diff) - (Linear_Diff)/2

# Make some tables and plots of the data.
print(RUL_Pred.head())
print("-"*100)
print(RUL_Pred.dtypes)
print("-"*100)
print(RUL_Pred.describe())
print("-"*100)

sns.boxplot(data=RUL_Pred[["LSTM Loss", "Linear Loss"]])
plt.savefig("Figures/PairedTTest/BoxPlot.png")
plt.close()
# Looks like equal variances, but the means might not be equal.

# Lets do the paired t-test
model = stats.ttest_rel(RUL_Pred["LSTM Loss"], RUL_Pred["Linear Loss"])
print(model)
print("-"*100)
# Since the pvalue is above 0.05 we cannot reject the null hypothesis that the mean of the losses are the same.
# That is if the model assumptions hold

# The assumptions that must hold are Independence, Same variances and Normality

# The independence of the data is not something that we can check at this point as they have to do with the collection of the data.
# We do not see any problems related to independence during the data collection for this project

# We can already get an indication of the variances by looking at our boxplot - here they look fine.
# Alternatively we can use the formal levenes test.
levene = stats.levene(RUL_Pred["LSTM Loss"], RUL_Pred["Linear Loss"])
print(levene)
# The levenes test confirms what we saw on the boxplot - we have no reason to reject the assumption of equal variances.

# To test for normality we plot the data against the theoretical quantiles of a normal distribution in a qq-plot
sm.qqplot(RUL_Pred["LSTM Loss"] - RUL_Pred["Linear Loss"])
plt.savefig("Figures/PairedTTest/Qqplot.png")
plt.close()
# The qqplot seems a little off with one possible outlier

# Since the assumptions hold we can accept the null hypothesis that
# there is not a statistically significant difference at the 5% test level.
