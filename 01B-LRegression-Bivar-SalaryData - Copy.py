# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 00:18:39 2018
@filename: LinearRegression-Simple
@dataset: lr-data
@author: cyruslentin
"""

# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
# seaborn
import seaborn as sns
# utils
import utils

##############################################################
# Read Data 
##############################################################



# read dataset
print("\n*** Read Data ***")
df = pd.read_csv('./data/slr-salary-data.csv')
print("Done ...")

##############################################################
# Exploratory Data Analysis
##############################################################

# rows & cols
print("\n*** Rows & Cols ***")
print("Rows",df.shape[0])
print("Cols",df.shape[1])

# columns
print("\n*** Column Names ***")
print(df.columns)

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())

# info
print("\n*** Structure ***")
print(df.info())


##############################################################
# Dependent Variable 
##############################################################

# store dep variable  
# change as required
xVars = "YearsExperience"
yVars = "Salary"
print("\n*** Vars ***")
print("xVars:",xVars)
print("yVars:",yVars)


##############################################################
# Data Transformation
##############################################################

# drop cols
# serial cols
# identification cols
# nominal columns
# description columns
# change as required
print("\n*** Drop Cols ***")
df = df.drop('Ser', axis=1)
print("Done ...")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier row index 
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers if required

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# handle normalization if required

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required

# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.2f}'.format
print(df.corr())

# handle multi colinearity if required


##############################################################
# Visual Data Analysis
# https://seaborn.pydata.org/introduction.html#:~:text=Seaborn%20is%20a%20library%20for,explore%20and%20understand%20your%20data.
##############################################################

# check relation with corelation - heatmap
print("\n*** Heat Map ***")
plt.figure(figsize=(4,4))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)

# histograms
# https://www.qimacros.com/histogram-excel/how-to-determine-histogram-bin-interval/
# plot histograms
print('\n*** Histograms ***')
colNames = df.columns.tolist()
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# scatter plot - xVars v/s yVars
print('\n*** Scatterplot ***')
plt.figure()
sns.regplot(x=xVars, y=yVars, data=df, color= 'b')
plt.title(xVars + ' v/s ' + yVars)
plt.xlabel(xVars)
plt.ylabel(yVars)
# good practice
plt.show()

##############################################################
# Model Creation & Fitting And Prediction for Feature 
##############################################################

# all cols except dep var
print("\n*** Regression Data ***")
print(xVars)
print(yVars)

# regression summary for feature
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(df[xVars])
y = df[yVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# prepare data
print("\n*** Prepare Data ***")
X = df[xVars].values.reshape(-1,1)
y = df[yVars].values
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# now create linear regression model
print("\n*** Regression Model ***")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model)

# train model
print("\n*** Train Model ***")
model.fit(X,y)
print("Done ...")

# predict
print("\n*** Predict Data ***")
p = model.predict(X)
df['predict'] = p
print("Done ...")

##############################################################
# Model Evaluation
#https://towardsdatascience.com/which-evaluation-metric-should-you-use-in-machine-learning-regression-problems-20cdaef258e
##############################################################

# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
sns.regplot(data=df, x=yVars, y='predict', color='b', scatter_kws={"s": 10})
plt.show()

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(df[yVars], df['predict'])
print(mae)

# mse
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df[yVars], df['predict'])
print(mse)
   
# rmse 
# https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# scatter index (SI) is defined to judge whether RMSE is good or not. 
# SI=RMSE/mean of actual data 
# If SI is less than one, your estimations are acceptable.
# closer to zero the better
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
# also called normalised RMSE
# https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/
print('\n*** Scatter Index ***')
si = rmse/df[yVars].mean()
print(si)

# predict new data ... change as required
prd_X = np.array([[2.5],[3.5],[4.5],[5.5]])
print(prd_X.flatten())

prd_p = model.predict(prd_X)
print(prd_p)

# input
while (True):
    prdX = input("Input: ")
    if prdX == "":
        exit
    prdX = float(prdX)
    prdX = np.array([[prdX]])
    prdp = model.predict(prdX)
    print("Predict: ", prdp)

