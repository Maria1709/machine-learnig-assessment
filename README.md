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


# Lets create a subset of data to make plotting graphs more legible

CRIM per capita cime rate by town
RM average number of rooms per dwelling
B proportion of blacks by town
MEDV median value of owner ooccupied homes

cols = ['CRIM', 'RM', 'B', 'MEDV', 'LSTAT']
# First five columns and headings

CRIM	RM	B	MEDV	LSTAT
0	0.00632	6.575	396.90	24.0	4.98
1	0.02731	6.421	396.90	21.6	9.14
2	0.02729	7.185	392.83	34.7	4.03
3	0.03237	6.998	394.63	33.4	2.94
4	0.06905	7.147	396.90	36.2	5.33


# WE NOW START PLOTTING SOME DATA

# Lets compare the house prices against some of the subset of features

fig, ax = plt.subplots(1, 2)
sns.regplot('RM', 'MEDV', df, ax=ax[0],
scatter_kws={'alpha': 0.4})
sns.regplot('LSTAT', 'MEDV', df, ax=ax[1],
scatter_kws={'alpha': 0.4}) 


# RM and MEDV have the closest shape to normal distributions.
b: AGE is skewed to the left and LSTAT is skewed to the right (this may seem counter intuitive but skew is defined in terms of where the mean is positioned in relation to the max).
c: For TAX, we find a large amount of the distribution is around 700. This is also evident from the scatter plots

![Image description](https://static.packt-cdn.com/products/9781789804744/graphics/37fa1155-b42e-4755-b859-5b12df9784fd.png)

# Lets compare the house prices against some of the subset of features

fig, ax = plt.subplots(1, 2)
sns.regplot('CRIM', 'MEDV', df, ax=ax[0],
scatter_kws={'alpha': 0.4})
sns.regplot('B', 'MEDV', df, ax=ax[1],
scatter_kws={'alpha': 0.4})

# Plot multiple data points

CHAS = df.CHAS.values
MEDV = df.MEDV.values
RM = df.RM.values
plt.plot(MEDV, 'r--', CHAS, 'gs', RM, 'g^')

# INFER

![Image description](http://onlinelifestyle.co/wp-content/uploads/2018/12/map-of-neighborhoods-boston-massachusetts.jpg)
# Lets take a look at the correlations between various data points in the Boston Housing dataset

We can see here that the correlation between proximity to the Charles River and the Median price is 0.175260 
Which indicates no real correlation between these data points
df.corr()



# CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV

CRIM	1.000000	-0.200469	0.406583	-0.055892	0.420972	-0.219247	0.352734	-0.379670	0.625505	0.582764	0.289946	-0.385064	0.455621	-0.388305
ZN	-0.200469	1.000000	-0.533828	-0.042697	-0.516604	0.311991	-0.569537	0.664408	-0.311948	-0.314563	-0.391679	0.175520	-0.412995	0.360445
INDUS	0.406583	-0.533828	1.000000	0.062938	0.763651	-0.391676	0.644779	-0.708027	0.595129	0.720760	0.383248	-0.356977	0.603800	-0.483725
CHAS	-0.055892	-0.042697	0.062938	1.000000	0.091203	0.091251	0.086518	-0.099176	-0.007368	-0.035587	-0.121515	0.048788	-0.053929	0.175260
NOX	0.420972	-0.516604	0.763651	0.091203	1.000000	-0.302188	0.731470	-0.769230	0.611441	0.668023	0.188933	-0.380051	0.590879	-0.427321
RM	-0.219247	0.311991	-0.391676	0.091251	-0.302188	1.000000	-0.240265	0.205246	-0.209847	-0.292048	-0.355501	0.128069	-0.613808	0.695360
AGE	0.352734	-0.569537	0.644779	0.086518	0.731470	-0.240265	1.000000	-0.747881	0.456022	0.506456	0.261515	-0.273534	0.602339	-0.376955
DIS	-0.379670	0.664408	-0.708027	-0.099176	-0.769230	0.205246	-0.747881	1.000000	-0.494588	-0.534432	-0.232471	0.291512	-0.496996	0.249929
RAD	0.625505	-0.311948	0.595129	-0.007368	0.611441	-0.209847	0.456022	-0.494588	1.000000	0.910228	0.464741	-0.444413	0.488676	-0.381626
TAX	0.582764	-0.314563	0.720760	-0.035587	0.668023	-0.292048	0.506456	-0.534432	0.910228	1.000000	0.460853	-0.441808	0.543993	-0.468536
PTRATIO	0.289946	-0.391679	0.383248	-0.121515	0.188933	-0.355501	0.261515	-0.232471	0.464741	0.460853	1.000000	-0.177383	0.374044	-0.507787
B	-0.385064	0.175520	-0.356977	0.048788	-0.380051	0.128069	-0.273534	0.291512	-0.444413	-0.441808	-0.177383	1.000000	-0.366087	0.333461
LSTAT	0.455621	-0.412995	0.603800	-0.053929	0.590879	-0.613808	0.602339	-0.496996	0.488676	0.543993	0.374044	-0.366087	1.000000	-0.737663
MEDV	-0.388305	0.360445	-0.483725	0.175260	-0.427321	0.695360	-0.376955	0.249929	-0.381626	-0.468536	-0.507787	0.333461	-0.737663	1.000000

# We can visually display these correlations in a heatmap
# Again we can see the color code indicatinf no strong positive or negative correlation between the data points
hm = sns.heatmap(df.corr())

![Image description](https://miro.medium.com/max/518/1*PyTZ8L1OLeIAa1HxjK_m5w.png)

