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
