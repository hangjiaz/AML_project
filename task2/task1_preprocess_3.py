import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso, Ridge
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold as VarThresh
from sklearn.covariance import MinCovDet as mcd
import scipy.linalg as linalg
from sklearn.impute import SimpleImputer as SI
#from sklearn.impute import IterativeImputer as II 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from helper_function import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

def data_preprocess():
	x_traindata = pd.read_csv("X_train.csv", header=0)
	y_traindata = pd.read_csv("y_train.csv", header=0)
	testdata = pd.read_csv("X_test.csv", header=0)
	
	#remove features that has more than 100 missing values
	# This subject to change: potentially information loss. 
	x_train = x_traindata.iloc[:,1:833].values
	x_test = testdata.iloc[:,1:833].values
	y_train = y_traindata.iloc[:,1].values
	testid = testdata.iloc[:,0].values
	null_cnt = x_traindata.isnull().sum(axis=0)
	null_cnt_np = null_cnt.values
	null_index = np.where(null_cnt_np > 300)

	#check test dataset
	#test_null_cnt = testdata.isnull().sum(axis=0)
	#test_null_cnt_np = test_null_cnt.values
	#test_null_index = np.where(test_null_cnt_np > 100)

	x_traindata_clean_missing = np.delete(x_train,(null_index),axis=1)
	x_test_clean_missing = np.delete(x_test,(null_index),axis=1)

	#data imputation(2 imputer can be used: Simple Imputor/Iterative Imputer)
	# fill method with simple imputation, not much difference between these 2 imputer
	imp_mean = SI(missing_values=np.nan, strategy='mean')
	x_train_complete = imp_mean.fit_transform(x_traindata_clean_missing)
	x_test_complete = imp_mean.fit_transform(x_test_clean_missing)

	# then try Variance inflation factor to remove
	# might not be a good idea to do so. Given that although prediction on validation set
	# will have 0.67 but actual submission yields negative result. 
	# vif_map = np.zeros(np.shape(x_train_complete)[1])
	# for i in range(x_train_complete.shape[1]):
	# 	vif = VIF(x_train_complete, i)
	# 	if (vif >= 5):
	# 		vif_map[i] = 1

	# print(np.count_nonzero(vif_map == 1))
	# vif_remove = np.where(vif_map == 1)
	# reduced_vif_train = np.delete(x_train_complete,(vif_remove),axis=1) 
	# reduced_vif_test = np.delete(x_train_complete,(vif_remove),axis=1)

	# normaliza data
	scaler = StandardScaler()
	x_train_norm = scaler.fit_transform(x_train_complete)
	x_test_norm = scaler.fit_transform(x_test_complete)
	# 
	selector = VarThresh(threshold = 0.01)
	selector.fit(x_train_norm)
	x_train_high_var = selector.transform(x_train_norm)
	x_test_high_var = selector.transform(x_test_norm)
	#print(np.shape(x_train_high_var))
	#print(np.shape(x_test_high_var))

	#correlation map to y 1st dimension
	corr_mat = np.zeros(np.shape(x_train_high_var)[1])
	#print(np.shape(corr_mat))
	for i in range(np.shape(x_train_high_var)[1]):
		corr = np.corrcoef(x_train_high_var[:,i], y_train)
		corr_mat[i] = corr[0,1]

	# correlation in second order: 
	x_train_2nd = np.square(x_train_high_var)
	corr_mat_2nd = np.zeros(np.shape(x_train_2nd)[1])
	#print(np.shape(corr_mat))
	for i in range(np.shape(x_train_2nd)[1]):
		corr_2nd = np.corrcoef(x_train_2nd[:,i], y_train)
		corr_mat_2nd[i] = corr_2nd[0,1]

	# correlation in third order: 
	x_train_3rd = np.power(x_train_high_var,3)
	corr_mat_3rd = np.zeros(np.shape(x_train_3rd)[1])
	#print(np.shape(corr_mat))
	for i in range(np.shape(x_train_3rd)[1]):
		corr_3rd = np.corrcoef(x_train_3rd[:,i], y_train)
		corr_mat_3rd[i] = corr_3rd[0,1]

	# log transformation correlation
	x_train_log = np.log(x_train_high_var)
	corr_mat_log = np.zeros(np.shape(x_train_log)[1])
	#print(np.shape(corr_mat))
	for i in range(np.shape(x_train_log)[1]):
		corr_log = np.corrcoef(x_train_log[:,i], y_train)
		corr_mat_log[i] = corr_log[0,1]

	#for now, keep the ones with absolute value larger than 0.1
	#as a matter of fact, log transformation doesn't matter that much
	low_corr = np.where(abs(corr_mat) < 0.1)
	low_corr_2nd = np.where(abs(corr_mat_2nd) < 0.1)
	low_corr_3rd = np.where(abs(corr_mat_3rd) < 0.1)
	low_corr_log = np.where(abs(corr_mat_log) < 0.4)
	print(np.shape(low_corr), np.shape(low_corr_2nd), np.shape(low_corr_3rd), np.shape(low_corr_log))
	#print(corr_mat_2nd[251], corr_mat_3rd[125], corr_mat_3rd[244], corr_mat_3rd[614], corr_mat_3rd[692], corr_mat_3rd[766])
	
	#reduction on first order
	reduced_matrix_train_1st = np.delete(x_train_high_var,(low_corr),axis=1)
	reduced_matrix_test_1st = np.delete(x_test_high_var, (low_corr), axis=1)
	#reduction on second order
	reduced_matrix_train_2nd = np.delete(x_train_high_var,(low_corr_2nd),axis=1)
	reduced_matrix_test_2nd = np.delete(x_test_high_var, (low_corr_2nd), axis=1)
	#reduction on third order
	reduced_matrix_train_3rd = np.delete(x_train_high_var,(low_corr_3rd),axis=1)
	reduced_matrix_test_3rd = np.delete(x_test_high_var, (low_corr_3rd), axis=1)
	#reduction on log transform
	#reduced_matrix_train_log = np.delete(x_train_high_var,(low_corr_log),axis=1)
	#reduced_matrix_test_log = np.delete(x_test_high_var, (low_corr_log), axis=1)
	#stack reduction together
	print(np.shape(reduced_matrix_train_1st))
	print(np.shape(reduced_matrix_train_2nd))
	print(np.shape(reduced_matrix_train_3rd))
	#print(np.shape(reduced_matrix_train_log))
	reduced_matrix_train = np.column_stack((reduced_matrix_train_1st, reduced_matrix_train_2nd))
	reduced_matrix_train = np.column_stack((reduced_matrix_train, reduced_matrix_train_3rd))
	#reduced_matrix_train = np.column_stack((reduced_matrix_train, reduced_matrix_train_log))

	reduced_matrix_test = np.column_stack((reduced_matrix_test_1st, reduced_matrix_test_2nd))
	reduced_matrix_test = np.column_stack((reduced_matrix_test, reduced_matrix_test_3rd))
	#reduced_matrix_test = np.column_stack((reduced_matrix_test, reduced_matrix_test_log))


	# for i in range(np.shape(x_high_var)[1]):
	# 	if high_corr == i:
	# 		reduced_matrix = np.append(reduced_matrix,x_high_var[:,i])
	#print(np.shape(reduced_matrix))


	# remove high correlation with each other
	# In fact, nothing worth to remove. 

	df_reduced = pd.DataFrame(reduced_matrix_train)
	df_corr_xx = df_reduced.corr(method = 'pearson')
	corr_xx = df_corr_xx.values
	high_corr_xx = np.where(abs(corr_xx)>0.95)
	#print(np.shape(high_corr_xx))	


	
	# reduced_x_train = reduced_matrix_train
	# reduced_x_test = reduced_matrix_test

	reduced_x_train = reduced_matrix_train
	reduced_x_test = reduced_matrix_test

	#Outlier Removal
	# 3 methods: kernelPCA, KNN, and IsoTree

	# This is KernelPCA method
	#transformer = KernelPCA(n_components=2, kernel='linear')
	#transformed_data = transformer.fit_transform(reduced_x_train)
	#x_clean, y_clean = remove_outlier(transformed_data,reduced_x_train, y_train,15)
	#kernel method is not optimistic
	
	reduced_full_mat = np.column_stack((reduced_x_train,y_train))
	
	# This is LocalOutlierFactor
	LOF = LocalOutlierFactor(n_neighbors=40, contamination=0.08)
	LOF.fit(reduced_full_mat)
	y_pred_local = LOF.fit_predict(reduced_full_mat)
	locations = np.where(y_pred_local == -1)


	#print(np.shape(y_pred_local))
	# This is IsoForest
	rng = np.random.RandomState(42)
	IsoTree = IsolationForest(max_samples=100,random_state=rng,contamination=0.08)
	IsoTree.fit(reduced_full_mat)
	y_pred_iso = IsoTree.predict(reduced_full_mat)
	locations = np.where(y_pred_iso == -1)

	x_clean = reduced_x_train
	y_clean = y_train
	for i in range(len(y_pred_local)-1,-1,-1):
		if ((y_pred_iso[i] == -1) and (y_pred_local[i] == -1)):
			x_clean = np.delete(x_clean, i, axis=0)
			y_clean = np.delete(y_clean, i, axis=0)

	print(np.shape(x_clean))
	print(np.shape(y_clean))


	return x_clean, y_clean, reduced_x_test, testid

# For testing preprocess purpose
#data_preprocess()





