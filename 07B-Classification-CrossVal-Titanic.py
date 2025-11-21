# -*- coding: utf-8 -*-
"""
@Created on Thu Mar 29 00:18:39 2018
@filename: Classification-CrossVal-Titanic
@dataset: titanic-train.csv & titanic-test.csv
@learnings: basic classification with basic checks and basic handling & cross val
@learnings: file name  not always -prd.csv || no clsVars column
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
plt.rcParams['figure.figsize'] = (10, 5)
# sns
import seaborn as sns
# util
import utils


##############################################################
# Read Data 
##############################################################

# read dataset
print("\n*** Read Data ***")
df = pd.read_csv('./data/titanic_train.csv')
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

# data types
print("\n*** Data Types ***")
print(df.dtypes)

# count of unique values
print("\n*** Unique Values ***")
print(df.apply(lambda x: x.nunique()))

# summary numeric cols
print("\n*** Summary Numeric ***")
dsTypes = df.dtypes
if "int" in dsTypes.tolist() or "float" in dsTypes.tolist():
    print(df.describe(include=np.number))
else:
    print("None ...")

# summary object cols
print("\n*** Summary AlphaNumeric ***")
dsTypes = df.dtypes
if "object" in dsTypes.tolist():
    print(df.describe(include=object))
else:
    print("None ...")

# head
print("\n*** Head ***")
print(df.head())

# info
print("\n*** Structure ***")
print(df.info())


##############################################################
# Class Variable & Counts
##############################################################

# store class variable  
# change as required
clsVars = "Survived"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

##############################################################
# Data Transformation
##############################################################

# drop cols
# identifiers
# nominals
# descriptors
# change as required
print("\n*** Drop Cols ***")
df = df.drop('PassengerId', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
df = df.drop('Embarked', axis=1)
print("Done ...")

# # transformations - convert object to float
# # change as required
# print("\n*** Transformations ***")
# # convert object to float
# colNames = ['xxx']
# for colName in colNames:
#     df[colName] = pd.to_numeric(df[colName], errors = "coerce")
#print("Done ...")

# get unique Sex names - convert string / categoric to numeric
print("\n*** Unique Sex - Categoric Alpha to Numeric ***")
print(df['Sex'].unique())
from sklearn import preprocessing
leSex = preprocessing.LabelEncoder()
df['Sex'] = leSex.fit_transform(df['Sex'])
print(df['Sex'].unique())

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
print('\n*** Handle Outliers ***')
df = utils.HandleOutliers(df, clsVars)
print("Done ...")

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required
print('\n*** Handle Zeros ***')
df = utils.HandleZeros(df, "Fare")
print("Done ...")

# drop col if all values are same
print("\n*** Same Value Cols Drop ***")
lDropCols = utils.SameValuesCols(df, clsVars, 1)
print(lDropCols)
if lDropCols != []:
    df = df.drop(lDropCols, axis=1)
print("Done ...")

# drop col if contains 100% unique values
print("\n*** Uniq Value Cols Drop ***")
lDropCols = utils.UniqValuesCols(df, clsVars, 1)
print(lDropCols)
if lDropCols != []:
    df = df.drop(lDropCols, axis=1)
print("Done ...")

# drop col if more than 50% null values
print("\n*** Null Value Cols Drop ***")
lDropCols = utils.NullValuesCols(df, clsVars, 0.50)
print(lDropCols)
if lDropCols != []:
    df = df.drop(lDropCols, axis=1)
print("Done ...")

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required
print('\n*** Handle Nulls ***')
df = utils.HandleNullsWithMean(df, clsVars)
print("Done ...")

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# # handle normalization if required
# print('\n*** Normalize Data ***')
# df = utils.NormalizeData(df, clsVars)
# print('Done ...')


##############################################################
# Visual Data Anlytics
##############################################################

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
colNames.remove(clsVars)
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# histograms
# plot histograms
print("\n*** Histogram Plot ***")
colNames = df.columns.tolist()
colNames.remove(clsVars)
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()
    
# class count plot
# change as required
vMaxCats = 10
colNames = df.columns.tolist()
colNames.remove(clsVars)
print("\n*** Distribution Plot ***")
bFlag = False
for colName in colNames:
    if len(df[colName].unique()) > vMaxCats:
        continue
    plt.figure()
    sns.countplot(data=df, x=colName,label="Count")
    plt.title(colName)
    plt.show()
    bFlag = True
if bFlag==False:
    print("No Categoric Variables Found")

# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby(clsVars).size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(data=df, x=clsVars,label="Count")
plt.title('Class Variable')
plt.show()


################################
# Classification 
# set X & y
###############################

# split into data & target
print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(clsVars)
print(allCols)
X = df[allCols].values
y = df[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
print(y[0:4])


################################
# Classification - init models
###############################

# original
# import all model & metrics
print("\n*** Importing Models ***")
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB
print("Done ...")

# create a list of models so that we can use the models in an iterstive manner
print("\n*** Creating Models ***")
lModels = []
lModels.append(('SVM-Clf', SVC(random_state=707)))
lModels.append(('RndFrst', RandomForestClassifier(random_state=707)))
lModels.append(('KNN-Clf', KNeighborsClassifier()))
lModels.append(('LogRegr', LogisticRegression(random_state=707)))
lModels.append(('DecTree', DecisionTreeClassifier(random_state=707)))
lModels.append(('GNBayes', GaussianNB()))
for vModel in lModels:
    print(vModel)
print("Done ...")


################################
# Classification - cross validation
###############################

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModNames = []
xvAccuracy = []
xvSDScores = []
print("Done ...")

# cross validation
from sklearn import model_selection
#print("\n*** Cross Validation ***")
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # select xv folds
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=707)
    # actual corss validation
    cvAccuracy = model_selection.cross_val_score(oModelObj, X, y, cv=kfold, scoring='accuracy')
    # prints result of cross val ... scores count = lfold splits
    print(vModelName,":  ",cvAccuracy)
    # update lists for future use
    xvModNames.append(vModelName)
    xvAccuracy.append(cvAccuracy.mean())
    xvSDScores.append(cvAccuracy.std())
    
# cross val summary
print("\n*** Cross Validation Summary ***")
# header
msg = "%10s: %10s %8s" % ("Model   ", "xvAccuracy", "xvStdDev")
print(msg)
# for each model
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%10s: %5.7f %5.7f" % (xvModNames[i], xvAccuracy[i], xvSDScores[i])
    print(msg)

# find model with best xv accuracy & print details
print("\n*** Best XV Accuracy Model ***")
xvIndex = xvAccuracy.index(max(xvAccuracy))
print("Index      : ",xvIndex)
print("Model Name : ",xvModNames[xvIndex])
print("XVAccuracy : ",xvAccuracy[xvIndex])
print("XVStdDev   : ",xvSDScores[xvIndex])
print("Model      : ",lModels[xvIndex])

# Without Optimization
# *** Cross Validation Summary ***
#   Model   : xvAccuracy xvStdDev
#    SVM-Clf: 0.6666876 0.0300197
#    RndFrst: 0.8125604 0.0162293
#    KNN-Clf: 0.7047894 0.0219311
#    LogRegr: 0.7912623 0.0116737
#    DecTree: 0.7856255 0.0110462
#    GNBayes: 0.7856632 0.0182113
# *** Best XV Accuracy Model ***
# Index      :  1
# Model Name :  RndFrst
# XVAccuracy :  0.812560416797439
# XVStdDev   :  0.0162292715727876
# Model      :  ('RndFrst', RandomForestClassifier(random_state=707))

# With Outlier Handling
# *** Cross Validation Summary ***
#   Model   : xvAccuracy xvStdDev
#    SVM-Clf: 0.6678237 0.0228470
#    RndFrst: 0.8102944 0.0241477
#    KNN-Clf: 0.7048145 0.0210471
#    LogRegr: 0.7867679 0.0075982
#    DecTree: 0.7777603 0.0151392
#    GNBayes: 0.7755759 0.0230788
# *** Best XV Accuracy Model ***
# Index      :  1
# Model Name :  RndFrst
# XVAccuracy :  0.8102943945766118
# XVStdDev   :  0.024147673199223414
# Model      :  ('RndFrst', RandomForestClassifier(random_state=707))


################################
# Classification 
# Split Train & Test
###############################

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.2, random_state=707)

# shapes
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))


################################
# Classification- Create Model
###############################

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[xvIndex])
print("Model   :", lModels[xvIndex]) 

# classifier object
# select model with best acc
print("\n*** Classfier Object ***")
model = lModels[xvIndex][1]
print(model)
# fit the model
model.fit(X_train,y_train)
print("Done ...")


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

# classifier object
print("\n*** Predict Test ***")
# predicting the Test set results
p_test = model.predict(X_test)            # use model ... predict
print("Done ...")

# accuracy
from sklearn.metrics import accuracy_score
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_test, p_test)*100
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
from sklearn.metrics import confusion_matrix
print("\n*** Confusion Matrix - Original ***")
cm = confusion_matrix(y_test, y_test)
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
print("\n*** Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_test, p_test)
print(cm)

# Without Optimization
# *** Accuracy ***
# 82.68156424581005
# *** Confusion Matrix - Original ***
# [[110   0]
#  [  0  69]]
# *** Confusion Matrix - Predicted ***
# [[99 11]
#  [20 49]]

# With Outlier Handling <TO BE USED>
# *** Accuracy ***
# 82.68156424581005
# *** Confusion Matrix - Original ***
# [[110   0]
#  [  0  69]]
# *** Confusion Matrix - Predicted ***
# [[99 11]
#  [20 49]]

# classification report
from sklearn.metrics import classification_report
print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)

# make dftest
# only for show
# not to be done in production
print("\n*** Recreate Test ***")
dfTest =  pd.DataFrame(data = X_test)
dfTest.columns = allCols
dfTest[clsVars] = y_test
dfTest['Predict'] = p_test
#dfTest[clsVars] = le.inverse_transform(dfTest[clsVars])
#dfTest['Predict'] = le.inverse_transform(dfTest['Predict'])
print("Done ...")

################################
# Final Prediction
# Create model Object from whole data
# Read .prd file
# Predict Species
# Confusion matrix with data in .prd file
###############################

# classifier object
print("\n*** Classfier Object ***")
model = lModels[xvIndex][1]
print(model)
# fit / train the model
model.fit(X,y)
print("Done ...")

# read dataset
print("\n*** Predict Data - Read ***")
dfp = pd.read_csv('./data/titanic_test.csv')
print(dfp.info())

print("\n*** Drop Cols ***")
dfp = dfp.drop('PassengerId', axis=1)
dfp = dfp.drop('Name', axis=1)
dfp = dfp.drop('Ticket', axis=1)
dfp = dfp.drop('Cabin', axis=1)
dfp = dfp.drop('Embarked', axis=1)
print("Done ...")

# convert string / categoric to numeric
print("\n*** Predict Data - Class Vars ***")
#print(dfp[clsVars].unique())
print("N/A ...")

# change as required ... same transformtion as done for main data
print("\n*** Predict Data - Transformation ***")
# convert string / categoric to numeric
print(dfp['Sex'].unique())
dfp['Sex'] = leSex.transform(dfp['Sex'])
print(df['Sex'].unique())
print("Done ...")

# check outlier count
print('\n*** Predict Data - Outlier Count ***')
print(utils.OutlierCount(dfp))

# check outlier values
print('\n*** Predict Data - Outlier Values ***')
print(utils.OutlierValues(dfp))

# handle outliers if required
print('\n*** Predict Data - Handle Outliers ***')
dfp = utils.HandleOutliers(dfp)
print("Done ...")

# check zeros
print('\n*** Predict Data - Columns With Zeros ***')
print((dfp==0).sum())

# handle zeros if required
print('\n*** Predict Data - Handle Zeros ***')
dfp = utils.HandleZeros(dfp, "Fare")
print("Done ...")

# check nulls
print('\n*** Predict Data - Columns With Nulls ***')
print(dfp.isnull().sum()) 
print("Done ... ")

# check nulls
print('\n*** Predict Data - Handle Nulls ***')
for colName in allCols:
    if ((dfp[colName].isnull()).sum() > 0):
        dfp[colName] = dfp[colName].fillna(df[colName].mean())
print("Done ... ")

# # handle normalization if required
# # if normalization is done with main data 
# then must be done with predict data
# print('\n*** Predict Data - Normalize Data ***')
# df = utils.NormalizeData(df, clsVars)
# print('Done ...')

# split into data & outcome
print("\n*** Data For Prediction - X & y Split ***")
print(allCols)
print(clsVars)
X_pred = dfp[allCols].values
#y_pred = dfp[clsVars].values
print(X_pred)
#print(y_pred)

# predict from model
print("\n*** Prediction ***")
p_pred = model.predict(X_pred)
# actual
#print("Actual")
#print(y_pred)
# predicted
print("Predicted")
print(p_pred)

# update data frame
print("\n*** Update Predict Data ***")
dfp['Predict'] = p_pred
#dfp[clsVars] = le.inverse_transform(dfp[clsVars])
#dfp['Predict'] = le.inverse_transform(dfp['Predict'])
print("Done ...")

# show predicted values
print("\n*** Print Predict Data ***")
for idx in dfp.index:
     print(idx, "\t", dfp['Predict'][idx])
print("Done ... ")



