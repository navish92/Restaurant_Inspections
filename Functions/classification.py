"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Author(s): @Andrew Auyeung, @Julia Qiao, @Navish Agarwal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV,  StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score, make_scorer, classification_report, confusion_matrix
from sklearn.metrics import log_loss, precision_score, recall_score, accuracy_score


def classification_common_model (X_train_val, y_train_val, model, oversampler = None, scaler = None, \
                          threshold = 0.5, return_type = None):
    
    """
    Function does a kfold=5 split, scales & oversamples the data if requested, 
    fits the transformed data on the model specified & prints the average of the 5-Fold along various metrics.
    
    Inputs:
    X_train_val: Features to be used
    Y_train_val: Target variable
    model: Classification model to be used
    oversampler: Oversampling technique
    scalar: Scaling method to be used
    threshold: probability point at which prediction should be put into a specific class
    return_type: If the function should return a dictionary with all values; the scores only get printed.
    
    Outputs:
    Using data from the last of the 5 KFolds, returns a dictionary containing the fitted model, data used for validation,
    and prediction probabilities. Also returns the mean train, validaton, precision, recall, roc_auc and logloss scores
    along with the classification report.
    
    """
    
    #use stratified kfold to splice up train-val into train and val
    
    skfold = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)
    skfold.get_n_splits(X_train_val, y_train_val)
    
    #create a list to store our scores
    train_scores= []
    val_scores= []
    precision_scores= []
    recall_scores= []
    roc_auc_scores = []
    logloss_scores = []
    
    #fit and score model on each fold
    for train, val in skfold.split(X_train_val, y_train_val):
        
        #set up train and val for each fold
        X_train, X_val = X_train_val.iloc[train], X_train_val.iloc[val]
        y_train, y_val = y_train_val.iloc[train], y_train_val.iloc[val]
        
        #oversample train data
        if oversampler:
            X_train, y_train = oversampler.fit_sample(X_train, y_train)

        #Scale data
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
                
        #fit model
        model.fit(X_train, y_train)
        
        #make prediction using y-val
        #try-except used incase model.predict_proba throws an error
        try:
            y_pred_proba = model.predict_proba(X_val)[:,1]
            y_pred = y_pred_proba > threshold
        except:
            y_pred_proba = np.zeros(len(y_val))
            y_pred = model.predict(X_val)
        
        #append scores onto list
        train_scores.append(accuracy_score(y_train,  model.predict(X_train)))
        val_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, average='binary'))
        recall_scores.append(recall_score(y_val, y_pred, average='binary'))
        roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        logloss_scores.append(log_loss(y_val, y_pred_proba))
        
    #find the means for our scores
    mean_train = np.mean(train_scores)
    mean_val = np.mean(val_scores)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    roc_auc = np.mean(roc_auc_scores)
    logloss = np.mean(logloss_scores)
    
    #print our mean accuracy score, our train/test ratio precision, and recall
    print(f'Scores fit on {model}')
    print(f'Accuracy: {mean_val:.2f}')
    print(f'Train/Test ratio: {(mean_train)/(mean_val):.2f}')
    
    print(f'Precision: {precision:.2f}')
    print(f'RECALL: {recall:.2f}')
    print(f'Log Loss: {logloss:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')
    print(classification_report(y_val,y_pred))
    print('-----')      
    
    if return_type:
        return {'model':model, 'X_val':X_val, 'y_pred_proba':y_pred_proba, 'mean_train':mean_train, \
                'mean_val': mean_val, 'precision':precision, 'recall':recall, 'roc_auc':roc_auc, \
                'logloss':logloss, 'classification_report': classification_report(y_val,y_pred) }
    else:
        return None

def knn_cv(X_train_val, y_train_val, cv=5, verbose = False, scorer=make_scorer(fbeta_score, beta = 1.0)):
    """
    Fits and trains the KNeighborsClassifier with a GridSearchCV to find the best parameters
    Inputs:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (object): CrossValidation argument, Default 5-Fold         
        scorer: scoring metric to be used; Defaults to F1
    Outputs:
        model (estimator): best estimator with highest score (based on scorer provided)
    """
    
    pipe = imbPipeline(steps=[
        ('sample', RandomOverSampler()), 
        ('scaler', StandardScaler()), 
        ('knn', KNeighborsClassifier())
        ])
    params = [{'knn__n_neighbors': range(2,25), 'knn__p': [1, 2]}]

    model = GridSearchCV(pipe, params, cv=cv, n_jobs=-1, scoring=scorer, verbose=verbose)
    model.fit(X_train_val, y_train_val)
    return model

def logreg_cv(X_train_val, y_train_val, cv=5, scorer=make_scorer(fbeta_score, beta = 1.0), verbose=False):
    
    """
    Fits and trains a Logistic Regression model with GridSearchCV to find the best parameters
    Inputs:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (object): CrossValidation argument, Default 5-Fold
        scorer: scoring metric to be used; Defaults to F1
    Outputs:
        model (estimator): best estimator with highest score (based on scorer provided)
    """
    
    params = [{
        'logreg__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
        }]
    pipe = imbPipeline(steps=[
        ('sample', RandomOverSampler()),
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression())
        ])
    model = GridSearchCV(pipe, params, cv=cv, n_jobs=-1, scoring=scorer, verbose=verbose)
    model.fit(X_train_val, y_train_val)
    return model

def rf_cv(X_train_val, y_train_val, cv=5, scorer=make_scorer(fbeta_score, beta = 1.0), verbose=False):
    """
    Fits and trains a Random Forest model with GridSearchCV to find the best parameters
    Input:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (object): Default 5-Fold CrossValidation
        scorer: scoring metric to be used; Defaults to F1
    Outputs:
        model (estimator): best estimator with highest score (based on scorer provided)
    """
    
    rf = RandomForestClassifier()
    params = [{
        'rf__n_estimators': range(50, 450, 50),
        'rf__max_depth': [2, 5, 7], 
        'rf__min_samples_split': [5, 10, 20, 25],
        'rf__max_features': ['sqrt'],
        'rf__criterion': ['gini'], 
        }]

    pipe = imbPipeline(steps=[('sample', RandomOverSampler()), 
                              ('scaler', StandardScaler()), 
                              ('rf', RandomForestClassifier())])
    model = GridSearchCV(pipe, params, cv=cv, n_jobs=-1, scoring=scorer, verbose=verbose)
    model.fit(X_train_val, y_train_val)
    return model

