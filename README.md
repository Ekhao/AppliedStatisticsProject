# AppliedStatisticsProject
A repository for my project in the 02935 - Introduction to Applied Statistics for PhD Students at DTU

The root folder contains my project description and data for the 02935 Introduction to Applied Statistics 2022 at DTU.

The data is output from two predictive maintenance models predicting the remaining useful life of turbofan engines from the CMAPSSData data set. (https://ti.arc.nasa.gov/c/6/)
The actual remaining useful life of the turbofan engines is also included in the data.

The data is formatted as 3 rows and 100 columns.
Each column is a remaining useful life (real or prediction) for a simulated turbofan engine.
The first row contains the actual remaining useful life of the simulated engines taken from the CMAPSSData data set.
The second row contains predictions made using a LSTM Neural Network Model taken from kaggle. (https://www.kaggle.com/aysenur95/cmapss-damage-propagation-modeling)
The third row contains predictions made using a linear regression model taken from the following github page. (https://github.com/kpeters/exploring-nasas-turbofan-dataset)
