# -*- coding: utf-8 -*-
"""
Machine Learning for Engineering Sciences
Final Project Code
Author: Eli Borrin with Thomas Bausman
"""
#%% 
# Imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score

#%% Loading Data

# Clinical 
clinical_full = pd.read_csv("sample.group",
                 sep="\t",
                 header=0)

clinical = clinical_full.iloc[:,[0,1]]
clinical_disease = clinical_full.iloc[:,[0,1,5]]

# print(clinical)

# Species Data
species = pd.read_csv("mpa4_species.profile",
                 sep="\t",
                 header=0)
#print(species)
species = species.T

# Make the first row after transposition the header
species.columns = species.iloc[0]
species = species.drop(species.index[0])

# Reset index and make it the first column
species = species.reset_index()

# Rename the index column to something meaningful (e.g. 'PatientID')
species = species.rename(columns={'index': 'ID'})


#%% Temp Data Splitting
# Quantifying samples per disease
diseases = clinical_full.value_counts("LevelA")


# Function to return dataframe of just a desired disease 
def get_disease(df, column_name, target_class):
    """
    Return a subset of the DataFrame containing only rows of the desired class.

    """
    subset = df[df[column_name] == target_class].copy()
    return subset

immune = get_disease(clinical_disease, 'LevelA', 'immune disease')

#%% Data Pre-Processing 

# Encoding control and disease as 0 and 1 respectively and rename Sample as name to match species set
def encode_control_disease(clinical, label = 'Group'):
    clinical[label] = clinical[label].map({'Control': 0, 'Disease': 1})
    
    clinical = clinical.rename(columns={'Sample': 'ID'})
    return clinical

# Test/Training data function
def split_dataframe(df, test_size = 0.1):
    """
    Split a df into training and test subsets.

    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): The proportion of the DataFrame to include in the test set 
                           (e.g., 0.2 means 20% test data).

    Returns:
        train_df (pd.DataFrame): The training subset of the DataFrame.
        test_df (pd.DataFrame): The test subset of the DataFrame.
    """
    # Validate inputs
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a decimal between 0 and 1.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # Randomly shuffle the dataframe
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the split index
    test_count = int(len(df) * test_size)

    # Split the DataFrame
    test_df = shuffled_df.iloc[:test_count]
    train_df = shuffled_df.iloc[test_count:]

    return train_df, test_df

# Encoding
clinical_encoded = encode_control_disease(clinical)

# Splitting
clinical_train, clinical_test = split_dataframe(clinical_encoded)


#%% Model Functions

# Linear Regression Model

def linear_regression_model(clinical_train, clinical_test, species, id_col='ID', label_col='Group'):
    """
    Build and evaluate a linear regression model to classify patients as control/disease.

    Parameters
    
    clinical_train : pd.DataFrame
        Training subset of clinical data (already encoded: Control=0, Disease=1).
    clinical_test : pd.DataFrame
        Test subset of clinical data (already encoded: Control=0, Disease=1).
    species : pd.DataFrame
        DataFrame containing microbiome species abundances per patient.
    id_col : str
        Column name for patient ID present in both dataframes.
    label_col : str
        Column name for the encoded class label in clinical data.

    Returns
   
    model : LinearRegression
        Trained linear regression model.
    results : dict
        Model performance metrics (R², MSE, and classification accuracy).
    """

    # Merge species features with training and test clinical data
    train_df = pd.merge(clinical_train, species, on=id_col)
    test_df = pd.merge(clinical_test, species, on=id_col)

    # Separate features (X) and labels (y)
    X_train = train_df.drop(columns=[id_col, label_col])
    y_train = train_df[label_col]
    X_test = test_df.drop(columns=[id_col, label_col])
    y_test = test_df[label_col]

    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Convert continuous predictions to 0/1 class labels using threshold 0.5
    y_pred_class = (y_pred >= 0.5).astype(int)

    # Compute metrics
    results = {
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred_class)
    }

    return model, results

#linear_model, linear_results = linear_regression_model(clinical_train, clinical_test, species)


# Logistic Regression: Better for binary decisions
def logistic_regression_model(clinical_train, clinical_test, species, id_col='ID', label_col='Group'):
    """
    Build and evaluate a logistic regression model to classify patients as control/disease.

    Parameters
    
    clinical_train : pd.DataFrame
        Training subset of clinical data (labels encoded as 0/1).
    clinical_test : pd.DataFrame
        Test subset of clinical data (labels encoded as 0/1).
    species : pd.DataFrame
        DataFrame containing microbiome species abundances per patient.
    id_col : str
        Column name for patient ID present in both dataframes.
    label_col : str
        Column name for the encoded class label in clinical data.

    Returns
    
    model : LogisticRegression
        Trained logistic regression model.
    results : dict
        Dictionary containing classification metrics (Accuracy and ROC-AUC).
    """

    # Merge species features with training and test clinical data
    train_df = pd.merge(clinical_train, species, on=id_col)
    test_df = pd.merge(clinical_test, species, on=id_col)

    # Separate features (X) and labels (y)
    X_train = train_df.drop(columns=[id_col, label_col])
    y_train = train_df[label_col]
    X_test = test_df.drop(columns=[id_col, label_col])
    y_test = test_df[label_col]

    # Initialize and train logistic regression
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # Predict class labels and probabilities
    y_pred_class = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1 (Disease)

    # Compute metrics
    results = {
        'Accuracy': accuracy_score(y_test, y_pred_class),
        'ROC-AUC': roc_auc_score(y_test, y_pred_prob)
    }

    return model, results

#log_model, log_results = logistic_regression_model(clinical_train, clinical_test, species)