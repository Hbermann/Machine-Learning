#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:43:52 2023

@author: heatherbermann
"""

import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import OneHotEncoder
import joblib

import warnings # simply for ignoring all those deprecated warning messages
warnings.filterwarnings("ignore", category=FutureWarning)

#%% Getting the data

#read the training data set
all_df = pd.read_excel('trainDataset.xls')

#drop the ID column as it isn't used in training
all_df.drop('ID', inplace = True, axis=1)

#fill all null values as NaN
#all_df.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
all_df =all_df.replace(999, np.nan)

# Count number of NaN values per column
for column in all_df.columns:
    if (all_df[column].isnull().sum() > 0):
        print(column + ":" + str(all_df[column].isnull().sum()))
        
#%% Before we do any pre-processing, we want to split our dataset so that there is no data leakage from the test set into the training set. 

features_df = all_df.drop(["pCR (outcome)", "RelapseFreeSurvival (outcome)"], axis=1)
target_df = all_df[["pCR (outcome)", "RelapseFreeSurvival (outcome)"]]

# train test split
X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.25, random_state=0)

# print out shape of dataframes to make sure it worked okay
print("X_train.shape = ", X_train.shape)
print("X_test.shape = ", X_test.shape)
print("y_train.shape = ", y_train.shape)
print("y_test.shape = ", y_test.shape)


def calculateDistance(observation, missing, output_indicies):
    observation = observation.to_numpy()
    missing = missing.to_numpy()
    # print("size:" + str(observation.size))
    return absolute_distance(observation, missing, output_indicies)
    
def absolute_distance(observation, missing, output_indicies):
    sum = 0
    for x in range(0,observation.size-1):
        if(not(np.isnan(missing[x])) and not(x in output_indicies)): # do not calculate with outputs or null entries
            sum +=np.abs(observation[x] - missing[x])
    return sum
    
# df needs to not have missing values in itself
def impute_missing_value (df, missing, n_neighbours, output_indicies):
    distance = np.full(n_neighbours, np.inf)
    index = np.empty(n_neighbours)
    for x in range(0, len(df.index)):
        # only add to distance array if there is a smaller value
        
        dist = calculateDistance(df.iloc[x], missing, output_indicies)
        for y in range(0, n_neighbours):
            if(dist < distance[y]):
                distance[y] = dist
                index[y] = x
                break

    # for missing values in the missing observation, fill in using majority category
    # since no continuous data is missing

    booleans = np.isnan(missing)
    neighbours = df.iloc[index,:]
    for x in range(0,len(booleans)):
        if(booleans[x]):
            # calculate the majority class for that feature in nearest neighbours
            # find all the unique values for that feature
            # fill in at index x
            majority_x = neighbours.iloc[:,x].value_counts().idxmax()
            missing[x] = majority_x
    
    print(missing)
    return missing
    

def KNNimputation(df, n_neighbours, output_indicies):
    # Find all rows with nan values
    indices = df.index[df.isnull().any(axis=1)]
    nan_df = df[df.isnull().any(axis=1)]
    
    for x in range(0,len(nan_df)):    
        df.iloc[indices[x]] = impute_missing_value(df.drop([df.index[x]]), nan_df.iloc[x],n_neighbours, output_indicies)
    
    return df

k_neighbours = 15
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
output_cols = ["pCR (outcome)", "RelapseFreeSurvival (outcome)"]

# Training imputation
merged_train = pd.concat([X_train_normalised, y_train], axis = 1)
output_indicies = [merged_train.columns.get_loc(col) for col in output_cols]
knn_imputated_train_df = KNNimputation(merged_train, k_neighbours, output_indicies)


# Testing imputation
merged_test = pd.concat([X_test_normalised, y_test], axis = 1)
output_indicies = [merged_test.columns.get_loc(col) for col in output_cols]
knn_imputated_test_df = KNNimputation(merged_test, k_neighbours, output_indicies)


# Overwriting the existing train/test partioned dataframes with the new imputed dataframes
X_train = knn_imputated_train_df.drop(output_cols,axis=1)
y_train = knn_imputated_train_df.drop(merged_test.drop(output_cols, axis = 1), axis = 1)
X_test = knn_imputated_test_df.drop(output_cols, axis = 1)
y_test = knn_imputated_test_df.drop(merged_test.drop(output_cols, axis = 1), axis = 1)

knn_imputated_test_df.head()

#%% Preprossesing and normalisation

# define standard scaler model
scaler = StandardScaler()

# select categorical columns, they do not need to be scaled
categorical_columns = ["ER", "PgR", "HER2", "TrippleNegative", "ChemoGrade", "Proliferation", "HistologyType", "LNStatus", "TumourStage"]

# Seperate Categorical and non-cat columns
X_train_Cat = X_train[categorical_columns]
X_test_Cat = X_test[categorical_columns]
X_train_noCat = X_train.drop(categorical_columns, axis=1)
X_test_noCat = X_test.drop(categorical_columns, axis=1)

# normalise with fit_transform for the X_train data and just transform for the X_test data
normalised_X_train = pd.DataFrame(scaler.fit_transform(X_train_noCat), columns = X_train_noCat.columns)
normalised_X_test = pd.DataFrame(scaler.transform(X_test_noCat), columns = X_test_noCat.columns)

# export the scaler 
joblib.dump(scaler, 'Scaler.joblib')

# Merge normalised columns and categorical columns back together
X_train_Cat.reset_index(drop=True, inplace=True)
normalised_X_train.reset_index(drop=True, inplace=True)
X_train_normalised = pd.concat([X_train_Cat, normalised_X_train], axis=1)

# same for the test set
X_test_Cat.reset_index(drop=True, inplace=True)
normalised_X_test.reset_index(drop=True, inplace=True)
X_test_normalised = pd.concat([X_test_Cat, normalised_X_test], axis=1)

# print number of values missing in each partition of the train/test data
print("X_train_normalised missing data = ", X_train_normalised.isnull().sum().sum())
print("X_test_normalised missing data = ", X_test_normalised.isnull().sum().sum())
print("y_train missing data = ", y_train.isnull().sum().sum())
print("y_test missing data = ", y_test.isnull().sum().sum())
