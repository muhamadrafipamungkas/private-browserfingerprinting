# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:45:12 2020

@author: Katharina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__=="__main__":     
    data = pd.read_csv("../data/browsererkennung_features_tranco500.csv", sep=",")
    data = data.loc[:, data.isnull().sum() < 0.5*data.shape[0]] # Filter columns that contain 50% or more NaNs
    data = data.fillna(value=-1)
    
    # Only use completely measured URLs -> 5 + 5 runs
    counts = data['url'].value_counts()
    indices = [i for i,v in enumerate(counts) if v == 10]
    complete = counts[indices].index
    urls = list(complete)
    data = data[data['url'].isin(urls)]
    
    # Filter constant features
    data = data.loc[:,data.apply(pd.Series.nunique) != 1]
    
    # Create features, labels and groups
    groups = data['url'] 
    y = data['browser']
    X = data.drop(columns=['Unnamed: 0','url', 'run_id','browser']) # filter useless rows

    # Initialize result-lists to fill in while cross-validating
    recalls_chrome = list()
    precisions_chrome = list()
    recalls_ff = list()
    precisions_ff = list()
    accuracies = list()
    f1s_chrome = list()
    f1s_ff = list()
    finalFeatures = list()
    
    fold = 1 # For naming purposes  

    # Outer CV: split train and test set while looping so that all runs of a URL are either in training OR test set, never both
    for train_inds, test_inds in GroupKFold(n_splits = 10).split(X, y, groups):
        X_train, X_test, y_train, y_test = X.iloc[train_inds], X.iloc[test_inds], y.iloc[train_inds], y.iloc[test_inds]
    
        # Pipeline: scale -> select features -> fit model (RF)
        scaler = MinMaxScaler()
        kbest = SelectKBest() 
        base_estimator = RandomForestClassifier(random_state=0, n_jobs=5)
        pipeline = Pipeline([('scaler', scaler),('kbest', kbest), ('be', base_estimator)]) 
        
        # Parameter tuning via grid-search
        param_grid = {'kbest__score_func':[f_classif],'kbest__k': [1,2,4,8,16,32,64,128,256,len(X_train.columns)]}

        # Inner CV: like for outer CV split train and validation set so that all runs of a URL are either in training OR validation set, never both
        grid = GridSearchCV(pipeline,param_grid,cv=GroupKFold(n_splits=10).split(X_train, y_train, groups.iloc[train_inds]),refit=True, verbose=0).fit(X_train, y_train)

        # Classify with "winner" of grid-search
        classifier = grid.best_estimator_
        y_pred = classifier.predict(X_test)
        
        # Evaluate the prediction performance and append result-lists
        recalls_chrome.append(recall_score(y_test, y_pred, pos_label="chrome"))
        recalls_ff.append(recall_score(y_test, y_pred, pos_label="firefox"))
        precisions_chrome.append(precision_score(y_test, y_pred, pos_label="chrome"))
        precisions_ff.append(precision_score(y_test, y_pred, pos_label="firefox"))
        accuracies.append(accuracy_score(y_test, y_pred))
        f1s_chrome.append(f1_score(y_test, y_pred, pos_label="chrome"))
        f1s_ff.append(f1_score(y_test, y_pred, pos_label="firefox"))
        
        ########################################################
        ### Plotting feature importances for the outer folds ###
        ########################################################
        
        # The following is adjusted from https://stackoverflow.com/questions/40245277/visualize-feature-selection-in-descending-order-with-selectkbest
        
        ##################################################
        ### Ranking of Feature Selection (SelectKBest) ###
        ##################################################
        
        # Get the indices sorted by most important to least important
        indices = np.argsort(classifier.named_steps["kbest"].scores_)[::-1]
        
        # To get your top x feature names
        topx = 10
        features = []
        for i in range(topx):
            features.insert(0,X_train.columns[indices[i]])

        features = [feature.replace('protcol', 'protocol') for feature in features] # fix spelling error
        features = [feature.replace('between', 'btw') for feature in features] 
        
        # Now plot
        f = plt.figure(figsize=(4,3.0))
        plt.barh(features, classifier.named_steps["kbest"].scores_[indices[range(topx)]][::-1],zorder=2, edgecolor='black',color='#0d52a8', align='center')

        plt.grid(zorder=0, linestyle="--", alpha=0.15)
        plt.yticks(size='12.0')
        plt.xticks(size='12.0')
        
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel('ANOVA F-value',size='14.0')
        f.savefig("plots/fold"+str(fold)+"_KBEST.pdf", bbox_inches='tight')
             
        ##########################################
        ### Ranking of ML Model (RandomForest) ###
        ##########################################
        
        # Identify the selected features first, as now some features are dropped and the index differs from original index (i.e., differs from X.columns)
        support = classifier.named_steps["kbest"].get_support()
        X_selected = X.columns[support]
        importances = classifier.named_steps["be"].feature_importances_
        
        # Now repeat the same as above
        indices = np.argsort(importances)[::-1]
        
        features = []
        for i in range(topx):
            features.insert(0,X_selected[indices[i]])
        
        features = [feature.replace('protcol', 'protocol') for feature in features] # fix spelling error
        features = [feature.replace('between', 'btw') for feature in features] 
        
        f = plt.figure(figsize=(4,3.0))
        plt.barh(features, importances[indices[range(topx)]][::-1],zorder=2, edgecolor='black',color='#0d52a8', align='center')

        plt.grid(zorder=0, linestyle="--", alpha=0.15)
        plt.yticks(size='12.0')
        plt.xticks(size='12.0')
        
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel('MDI',size='14.0')
        plt.show()
        f.savefig("plots/fold"+str(fold)+"_RF.pdf", bbox_inches='tight')
                
        finalFeatures.append(X_selected) # For further inspection
        
        fold = fold + 1 # For naming purposes
        
    print("Avg. Accuracy: " + str(np.mean(accuracies)))
    print("Avg. Precision (Firefox): " + str(np.mean(precisions_ff)))
    print("Avg. Recall (Firefox): " + str(np.mean(recalls_ff)))
    print("Avg. F1 (Firefox): " + str(np.mean(f1s_ff)))
    print("Avg. Precision (Chrome): " + str(np.mean(precisions_chrome)))
    print("Avg. Recall (Chrome): " + str(np.mean(recalls_chrome)))
    print("Avg. F1 (Chrome): " + str(np.mean(f1s_chrome)))
    
    print("Std. Accuracy: " + str(np.std(accuracies)))
    print("Std. Precision (Firefox): " + str(np.std(precisions_ff)))
    print("Std. Recall (Firefox): " + str(np.std(recalls_ff)))
    print("Std. F1 (Firefox): " + str(np.std(f1s_ff)))
    print("Std. Precision (Chrome): " + str(np.std(precisions_chrome)))
    print("Std. Recall (Chrome): " + str(np.std(recalls_chrome)))
    print("Std. F1 (Chrome): " + str(np.std(f1s_chrome)))
        
    np.savetxt("results/accuracies.csv", accuracies, delimiter=",")
    np.savetxt("results/precisions_ff.csv", precisions_ff, delimiter=",")
    np.savetxt("results/recalls_ff.csv", recalls_ff, delimiter=",")
    np.savetxt("results/precisions_chrome.csv", precisions_chrome, delimiter=",")
    np.savetxt("results/recalls_chrome.csv", recalls_chrome, delimiter=",")
    
