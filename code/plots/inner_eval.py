# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:45:12 2020

@author: Katharina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

##########################################################################################################################
#### NOTE: This is just a quick and dirty script to plot the results of the hyperparameter-search/inner CV           #####
#### It basically manually implements the gridsearch, but does not actually evaluate the "winning" model             #####
#### So, this is just an adjusted copy-paste of the browserdetection.py script and should probably be merged into it #####
##########################################################################################################################

def lighten_color(color, amount=0.5):
    """
    from https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

if __name__=="__main__":     
    data = pd.read_csv("../../data/browsererkennung_features_tranco500.csv", sep=",")
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

    
    finalFeatures_all = list()
    
    fold = 1 # For naming purposes  

    # Outer CV: split train and test set while looping so that all runs of a URL are either in training OR test set, never both
    for train_inds, test_inds in GroupKFold(n_splits = 10).split(X, y, groups):
        # Initialize result-lists to fill in while cross-validating
        recalls_chrome = list()
        precisions_chrome = list()
        recalls_ff = list()
        precisions_ff = list()
        accuracies = list()
    
        
        sd_recalls_chrome = list()
        sd_precisions_chrome = list()
        sd_recalls_ff = list()
        sd_precisions_ff = list()
        sd_accuracies = list()
        X_train, X_test, y_train, y_test = X.iloc[train_inds], X.iloc[test_inds], y.iloc[train_inds], y.iloc[test_inds]
        nrFeatures = [1,2,4,8,16,32,64,128,256,len(X_train.columns)]
        
        for i in nrFeatures:
            # Pipeline: scale -> select features -> fit model (RF)
            scaler = MinMaxScaler()
            kbest = SelectKBest(k=i) 
            base_estimator = RandomForestClassifier(random_state=0, n_jobs=5)
            pipeline = Pipeline([('scaler', scaler),('kbest', kbest), ('be', base_estimator)]) 
            
            # Parameter tuning via grid-search
            inner_recalls_chrome = list()
            inner_precisions_chrome = list()
            inner_recalls_ff = list()
            inner_precisions_ff = list()
            inner_accuracies = list()
    
            # Inner CV: like for outer CV split train and validation set so that all runs of a URL are either in training OR validation set, never both
            for inner_train_inds, val_inds in GroupKFold(n_splits = 10).split(X_train, y_train, groups.iloc[train_inds]):
                X_inner_train, X_val, y_inner_train, y_val = X_train.iloc[inner_train_inds], X_train.iloc[val_inds], y_train.iloc[inner_train_inds], y_train.iloc[val_inds]
                pipeline.fit(X_inner_train, y_inner_train)
                y_pred = pipeline.predict(X_val)
                
                # Evaluate the prediction performance for each inner fold manually and append result-lists
                inner_recalls_chrome.append(recall_score(y_val, y_pred, pos_label="chrome"))
                inner_recalls_ff.append(recall_score(y_val, y_pred, pos_label="firefox"))
                inner_precisions_chrome.append(precision_score(y_val, y_pred, pos_label="chrome"))
                inner_precisions_ff.append(precision_score(y_val, y_pred, pos_label="firefox"))
                inner_accuracies.append(accuracy_score(y_val, y_pred))

            
            # Evaluate the prediction performance and append result-lists
            recalls_chrome.append(np.mean(inner_recalls_chrome))
            recalls_ff.append(np.mean(inner_recalls_ff))
            precisions_chrome.append(np.mean(inner_precisions_chrome))
            precisions_ff.append(np.mean(inner_precisions_ff))
            accuracies.append(np.mean(inner_accuracies))
            
            sd_recalls_chrome.append(np.std(inner_recalls_chrome))
            sd_recalls_ff.append(np.std(inner_recalls_ff))
            sd_precisions_chrome.append(np.std(inner_precisions_chrome))
            sd_precisions_ff.append(np.std(inner_precisions_ff))
            sd_accuracies.append(np.std(inner_accuracies))

            
        nrColors = 10

        template_color = '#0d52a8'
        colors = list()
        for i in range(nrColors):
            colors.append(lighten_color(template_color,0.125*(i+3)))
    
        sd1 = [sd_accuracies[0],sd_recalls_ff[0],sd_recalls_chrome[0],sd_precisions_ff[0],sd_precisions_chrome[0]]
        sd2 = [sd_accuracies[1],sd_recalls_ff[1],sd_recalls_chrome[1],sd_precisions_ff[1],sd_precisions_chrome[1]]
        sd4 = [sd_accuracies[2],sd_recalls_ff[2],sd_recalls_chrome[2],sd_precisions_ff[2],sd_precisions_chrome[2]]
        sd8 = [sd_accuracies[3],sd_recalls_ff[3],sd_recalls_chrome[3],sd_precisions_ff[3],sd_precisions_chrome[3]]
        sd16 = [sd_accuracies[4],sd_recalls_ff[4],sd_recalls_chrome[4],sd_precisions_ff[4],sd_precisions_chrome[4]]
        sd32 = [sd_accuracies[5],sd_recalls_ff[5],sd_recalls_chrome[5],sd_precisions_ff[5],sd_precisions_chrome[5]]
        sd64 = [sd_accuracies[6],sd_recalls_ff[6],sd_recalls_chrome[6],sd_precisions_ff[6],sd_precisions_chrome[6]]
        sd128 = [sd_accuracies[7],sd_recalls_ff[7],sd_recalls_chrome[7],sd_precisions_ff[7],sd_precisions_chrome[7]]
        sd256 = [sd_accuracies[8],sd_recalls_ff[8],sd_recalls_chrome[8],sd_precisions_ff[8],sd_precisions_chrome[8]]
        sdAll = [sd_accuracies[9],sd_recalls_ff[9],sd_recalls_chrome[9],sd_precisions_ff[9],sd_precisions_chrome[9]]
        
        mean1 = [accuracies[0],recalls_ff[0],recalls_chrome[0],precisions_ff[0],precisions_chrome[0]]
        mean2 = [accuracies[1],recalls_ff[1],recalls_chrome[1],precisions_ff[1],precisions_chrome[1]]
        mean4 = [accuracies[2],recalls_ff[2],recalls_chrome[2],precisions_ff[2],precisions_chrome[2]]
        mean8 = [accuracies[3],recalls_ff[3],recalls_chrome[3],precisions_ff[3],precisions_chrome[3]]
        mean16 = [accuracies[4],recalls_ff[4],recalls_chrome[4],precisions_ff[4],precisions_chrome[4]]
        mean32= [accuracies[5],recalls_ff[5],recalls_chrome[5],precisions_ff[5],precisions_chrome[5]]
        mean64 = [accuracies[6],recalls_ff[6],recalls_chrome[6],precisions_ff[6],precisions_chrome[6]]
        mean128 = [accuracies[7],recalls_ff[7],recalls_chrome[7],precisions_ff[7],precisions_chrome[7]]
        mean256 = [accuracies[8],recalls_ff[8],recalls_chrome[8],precisions_ff[8],precisions_chrome[8]]
        meanAll = [accuracies[9],recalls_ff[9],recalls_chrome[9],precisions_ff[9],precisions_chrome[9]]
        index = ['Accuracy', 'Recall\n(FF)', 'Recall\n(Chrome)', 'Precision\n(FF)', 'Precision\n(Chrome)']
        
        df = pd.DataFrame({'1': mean1,'2': mean2,'3': mean4,'4': mean8,'5': mean16,'6': mean32,'7': mean64,'8': mean128,'9': mean256,'10': meanAll}, index=index)
        sd_df = pd.DataFrame({'1': sd1,'2': sd2,'3': sd4,'4': sd8,'5': sd16,'6': sd32,'7': sd64,'8': sd128,'9': sd256,'10': sdAll}, index=index)
        
        ax = df.plot.bar(zorder=2,rot=0,color={"1": colors[0], "2": colors[1],"3": colors[2],"4": colors[3],"5": colors[4],"6": colors[5],"7": colors[6],"8": colors[7],"9": colors[8],"10": colors[9]}, width = 0.8,figsize=(10,1.90), yerr=sd_df.values.T)
        ax.grid(zorder=0,linestyle="--", alpha=0.15)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        leg = ax.legend(loc=(1.04,-0.35), title = "# Features",fontsize='12.0', frameon = False, borderpad = 0.0,labelspacing=0.0)
        leg.get_texts()[0].set_text('$2^0$')
        leg.get_texts()[1].set_text('$2^1$')
        leg.get_texts()[2].set_text('$2^2$')
        leg.get_texts()[3].set_text('$2^3$')
        leg.get_texts()[4].set_text('$2^4$')
        leg.get_texts()[5].set_text('$2^5$')
        leg.get_texts()[6].set_text('$2^6$')
        leg.get_texts()[7].set_text('$2^7$')
        leg.get_texts()[8].set_text('$2^8$')
        leg.get_texts()[9].set_text('All')
        leg.get_title().set_fontsize('14.0')
        leg._legend_box.sep = 3  
        
        plt.yticks(size='14.0')
        plt.xticks(rotation = 0, size='14.0')
        
        ax.figure.savefig("inner_eval"+str(fold)+".pdf", bbox_inches='tight')
        
        fold = fold + 1
