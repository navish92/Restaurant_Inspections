"""
Common Modeling & Graphing Functions
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sys import getsizeof
import datetime as dt
import ast
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter

def roc_auc_curve_plot(y_test,y_pred, title = "ROC curve"):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr,lw=2)
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])


    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title);
    print("ROC AUC score = ", roc_auc_score(y_test, y_pred))
    
    return None


    
def model_scaled (X_train_val, y_train_val, model, oversampler, return_type = None):
    
    """
    Function does a kfold=5 split, fits the provided data (after scaling & oversampling) on the model specified & outputs the average scores.
    Input: X & y dataframes, model to be used and oversampler to be used.
    Output:
    """
    #use stratified kfold to splice up on train-val into train and val
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
        
        #Scale data
        ss = StandardScaler()
        #fit transform X train
        X_train_scaled = ss.fit_transform(X_train)
        #transform X val
        X_val_scaled = ss.transform(X_val)
        
        #fit model
        model.fit(X_train_scaled, y_train)
        
        #make prediction using y-val
        y_pred =  model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:,1]
        
        #append scores onto list
        train_scores.append(model.score(X_train_scaled, y_train))
        val_scores.append(model.score(X_val_scaled, y_val))
        precision_scores.append(precision_score(y_val, y_pred, average='binary'))
        recall_scores.append(recall_score(y_val, y_pred, average='binary'))
        roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        logloss_scores.append(log_loss(y_val,y_pred_proba))
        
    #find the means for our scores
    mean_train = np.mean(train_scores)
    mean_val = np.mean(val_scores)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    roc_auc = np.mean(roc_auc_scores)
    logloss = np.mean(logloss_scores)
    
    #print our mean accuracy score, our train/test ratio precision, and recall
    print(f'Scores fit on {model}')
    print(f'Accuracy: {mean_val:.4f}')
    print(f'Train/Test ratio: {(mean_train)/(mean_val):.4f}')
    
    print(f'Precision: {precision:.4f}')
    print(f'RECALL: {recall:.4f}')
    print(f'Log Loss: {logloss:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(classification_report(y_val,y_pred))
    print('-----')
    
    
    if return_type:
        return {'model':model,'X_val':X_val_scaled,'y_pred_proba':y_pred_proba,'mean_train':mean_train,'precision':precision,'recall':recall,'roc_auc':roc_auc,'logloss':logloss}
    
def model_no_transformations (X_train_val, y_train_val, model, return_type = None):
    
    """
    Function does a kfold=5 split, fits the provided data (after scaling & oversampling) on the model specified & outputs the average scores.
    Input: X & y dataframes, model to be used and oversampler to be used.
    Output:
    """
    
    #use stratified kfold to splice up on train-val into train and val
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
        
               
        #fit model
        model.fit(X_train, y_train)
        
        #make prediction using y-val
        y_pred =  model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:,1]
        
        #append scores onto list
        train_scores.append(model.score(X_train, y_train))
        val_scores.append(model.score(X_val, y_val))
        precision_scores.append(precision_score(y_val, y_pred, average='binary'))
        recall_scores.append(recall_score(y_val, y_pred, average='binary'))
        roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        logloss_scores.append(log_loss(y_val,y_pred_proba))
        
    #find the means for our scores
    mean_train = np.mean(train_scores)
    mean_val = np.mean(val_scores)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    roc_auc = np.mean(roc_auc_scores)
    logloss = np.mean(logloss_scores)
    
    #print our mean accuracy score, our train/test ratio precision, and recall
    print(f'Scores fit on {model}')
    print(f'Accuracy: {mean_val:.4f}')
    print(f'Train/Test ratio: {(mean_train)/(mean_val):.4f}')
    
    print(f'Precision: {precision:.4f}')
    print(f'RECALL: {recall:.4f}')
    print(f'Log Loss: {logloss:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(classification_report(y_val,y_pred))
    print('-----')
    
    if return_type:
        return {'model':model,'X_val':X_val,'y_pred_proba':y_pred_proba,'mean_train':mean_train,'precision':precision,'recall':recall,'roc_auc':roc_auc,'logloss':logloss}