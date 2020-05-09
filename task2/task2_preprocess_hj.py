import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold 
import scipy.linalg as linalg
from sklearn.impute import SimpleImputer 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

def data_preprocess():
    x_traindata = pd.read_csv("X_train.csv", header=0)
    y_traindata = pd.read_csv("y_train.csv", header=0)
    testdata = pd.read_csv("X_test.csv", header=0)   	
    # read data
    x_train = x_traindata.iloc[:,1:1001].values
    x_test = testdata.iloc[:,1:1001].values
    y_train = y_traindata.iloc[:,1].values
    testid = testdata.iloc[:,0].values

    
    # downsample the majority class 1
    class1_idx = np.where(y_train == 1)[0]
    del_class1_idx = np.random.choice(class1_idx, size = len(class1_idx)-sum(y_train == 0), replace=False) 
    x_train = np.delete(x_train, (del_class1_idx), axis = 0) 
    y_train = np.delete(y_train, (del_class1_idx), axis = 0) 
    
       
    # feature selection 1: remove features with low variane
    selector = VarianceThreshold(threshold = 0)
    selector.fit(x_train)
    x_train_var = selector.transform(x_train)
    x_test_var = selector.transform(x_test)

	# normaliza data
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train_var)
    x_test_norm = scaler.fit_transform(x_test_var)

	
	#feature selection 2: remove features with low correlation with y
    corr_mat = np.zeros(x_train_norm.shape[1])
    for i in range(x_train_norm.shape[1]):
        corr = np.corrcoef(x_train_norm[:,i], y_train)
        corr_mat[i] = corr[0,1]
    low_corr = np.where(abs(corr_mat) < 0.05)
    reduced_train = np.delete(x_train_norm,(low_corr),axis=1)
    reduced_x_test = np.delete(x_test_norm , (low_corr), axis=1)

    #outlier removal using LOF and Isotree
    LOF = LocalOutlierFactor(n_neighbors=40, contamination=0.08)
    LOF.fit(reduced_train)
    y_pred_local = LOF.fit_predict(reduced_train)
    locations = np.where(y_pred_local == -1)
    
    rng = np.random.RandomState(42)
    IsoTree = IsolationForest(max_samples=100,random_state=rng,contamination=0.08)
    IsoTree.fit(reduced_train)
    y_pred_iso = IsoTree.predict(reduced_train)
    locations = np.where(y_pred_iso == -1)
    
    x_clean = reduced_train
    y_clean = y_train
    for i in range(len(y_pred_local)-1,-1,-1):
        if ((y_pred_iso[i] == -1) and (y_pred_local[i] == -1)):             
            x_clean = np.delete(x_clean, i, axis=0)
            y_clean = np.delete(y_clean, i, axis=0)
    
    return x_clean, y_clean, reduced_x_test, testid





