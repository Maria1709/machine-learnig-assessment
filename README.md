# Machine-Learning-Assessment

![Image description](https://ei.marketwatch.com/Multimedia/2016/10/12/Photos/ZH/MW-EX709_boston_20161012163949_ZH.jpg?uuid=053d9e12-90bc-11e6-9a13-00137241c023)




## This project applies basic machine learning concepts on data collected for housing prices in the Boston, Massachusetts area to predict the selling price of a new home.

## Software and Libraries

This project uses the following software and Python libraries:

Python 3.7

NumPy

Pandas

Scikit-learn

oS

Seaborn

Scipy.stats

Matplotlib.pyplot


## Introduction

This dataset that is used for this project was taken from the UCI Machine Learning Repository. The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusettes. For the purposes of this project, the following preoprocessing steps have been made to the dataset:

16 data points have an 'MDEV' value of 50.0. These data points likely contain missing or censored values and have been removed.
1 data point has an 'RM' value of 8.78. This data point can be considered an outlier and has been removed.
The features 'RM', 'LSTAT', 'PTRATIO', and 'MDEV' are essential. The remaining non-relevant features have been excluded.
The feature 'MDEV' has been multiplicatively scaled to account for 35 years of market inflation.

## Describe
We are asked to descibe the data set, we do this by
