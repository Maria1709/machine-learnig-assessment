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

We are asked to descibe the data set, we do this by:

## Take a look at the datset keys

 We can see the dataset has a number of keys which correspond to the data, target, features and a description.
 Then we take a look at each of the keys in turn starting with a description of the dataset
 We can see the various data points which are contained in the dataset. We aslo take a look at the feature names and the shape of the data set. The data set contains 506 rows and 13 columns

## Check the data set for any null values

We can see there are no blank fields in the dataset which may effect the statistics
We then take a look at the first 5 rows of data and the last 5 rows of data from the dataset.

## We can see there is no Price column as this is contained in the target variable
Lets add the column to the dataset
df['MEDV'] = boston.target
print(df.head())

## We then look at the maximum and the minimum price.
Max 50.0 Min 5.0

## Lets plot some of the statistic contained in the data set
The median value of properties is normally distributed except for a few outliers represented by the bell shaped curve. 

![Image description](https://miro.medium.com/max/866/1*1pVtTg-mmUbGRTkuXeTvkQ.png)


