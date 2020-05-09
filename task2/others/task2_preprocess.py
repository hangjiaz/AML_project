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
from sklearn.decomposition import KernelPCA, PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.feature_selection import SelectKBest,SelectFromModel
from sklearn.feature_selection import f_classif
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC, NuSVC, LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler as RUS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def data_preprocess_1():
	# loading data
	x_traindata = pd.read_csv("X_train.csv", header=0)
	y_traindata = pd.read_csv("y_train.csv", header=0)
	testdata = pd.read_csv("X_test.csv", header=0)

	x_train_raw = x_traindata.iloc[:,1:1001].values
	x_test = testdata.iloc[:,1:1001].values
	y_train_raw = y_traindata.iloc[:,1].values
	testid = testdata.iloc[:,0].values

	scaler = StandardScaler()
	x_stand_train = scaler.fit_transform(x_train_raw)
	x_stand_test = scaler.transform(x_test) 
	

	rus = RUS(random_state=0)
	x_resampled, y_resampled = rus.fit_resample(x_stand_train, y_train_raw)


	full_train = np.column_stack((x_resampled, y_resampled))
	#print(full_train[0,:])
	# Outlier Removal
	# This is LocalOutlierFactor
	LOF = LocalOutlierFactor(n_neighbors=40, contamination=0.08)
	LOF.fit(full_train)
	y_pred_local = LOF.fit_predict(full_train)
	locations = np.where(y_pred_local == -1)

	# This is IsoForest Method
	rng = np.random.RandomState(42)
	IsoTree = IsolationForest(max_samples=100,random_state=rng,contamination=0.08)
	IsoTree.fit(full_train)
	y_pred_iso = IsoTree.predict(full_train)
	locations = np.where(y_pred_iso == -1)

	for i in range(len(y_pred_local)-1,-1,-1):
		if ((y_pred_iso[i] == -1) and (y_pred_local[i] == -1)):
			x_clean = np.delete(x_resampled, i, axis=0)
			y_clean = np.delete(y_resampled, i, axis=0)

	lr = LogisticRegression(random_state=0,penalty='l2').fit(x_clean, y_clean)
	model = SelectFromModel(lr, prefit=True)
	x_train_norm  = model.transform(x_clean)
	x_test_norm  = model.transform(x_stand_test)

	#sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
	sel1 = SelectFromModel(LinearSVC(C=2, class_weight="balanced"))
	sel1.fit(x_clean, y_clean)

	valid_select_svc = sel1.get_support()

	sel2 = SelectFromModel(RandomForestClassifier(n_estimators=100))
	sel2.fit(x_clean, y_clean)

	valid_select_rf = sel2.get_support()

	sel3 = SelectFromModel(GradientBoostingClassifier(n_estimators=100))
	sel3.fit(x_clean, y_clean)

	valid_select_gbc = sel3.get_support()

	valid_x = np.zeros(np.shape(x_clean)[1])
	for i in range(len(valid_select_rf)):
		if(valid_select_rf[i]==False and valid_select_svc[i]==False and valid_select_gbc[i]==False):
			valid_x[i] = False
		else:
			valid_x[i] = True
	x_invalid = np.where(valid_x == False)	
	print(len(x_invalid))

	train_selected_x = np.delete(x_clean,(x_invalid),axis=1)
	test_selected_x = np.delete(x_stand_test,(x_invalid),axis=1)

	#x_train, x_val, y_train, y_val = train_test_split(x_train_norm, y_clean, test_size = 0.5)

	return train_selected_x, y_clean, test_selected_x,testid
	
def data_preprocess_2():
	# loading data
	x_traindata = pd.read_csv("X_train.csv", header=0)
	y_traindata = pd.read_csv("y_train.csv", header=0)
	testdata = pd.read_csv("X_test.csv", header=0)

	x_train_raw = x_traindata.iloc[:,1:1001].values
	x_test = testdata.iloc[:,1:1001].values
	y_train_raw = y_traindata.iloc[:,1].values
	testid = testdata.iloc[:,0].values

	scaler = StandardScaler()
	x_stand_train = scaler.fit_transform(x_train_raw)
	x_stand_test = scaler.transform(x_test) 

	rus = RUS(random_state=42)
	x_resampled, y_resampled = rus.fit_resample(x_stand_train, y_train_raw)


	full_train = np.column_stack((x_resampled, y_resampled))
	#print(full_train[0,:])
	# Outlier Removal
	# This is LocalOutlierFactor
	LOF = LocalOutlierFactor(n_neighbors=40, contamination=0.08)
	LOF.fit(full_train)
	y_pred_local = LOF.fit_predict(full_train)
	locations = np.where(y_pred_local == -1)

	# This is IsoForest Method
	rng = np.random.RandomState(42)
	IsoTree = IsolationForest(max_samples=100,random_state=rng,contamination=0.08)
	IsoTree.fit(full_train)
	y_pred_iso = IsoTree.predict(full_train)
	locations = np.where(y_pred_iso == -1)

	for i in range(len(y_pred_local)-1,-1,-1):
		if ((y_pred_iso[i] == -1) and (y_pred_local[i] == -1)):
			x_clean = np.delete(x_resampled, i, axis=0)
			y_clean = np.delete(y_resampled, i, axis=0)

	x_train, x_val, y_train, y_val = train_test_split(x_clean, y_clean, test_size = 0.2)


	return x_train,y_train,x_val,y_val,x_stand_test,testid	

def data_preprocess_3():
	# loading data
	x_traindata = pd.read_csv("X_train.csv", header=0)
	y_traindata = pd.read_csv("y_train.csv", header=0)
	testdata = pd.read_csv("X_test.csv", header=0)

	x_train_raw = x_traindata.iloc[:,1:1001].values
	x_test = testdata.iloc[:,1:1001].values
	y_train_raw = y_traindata.iloc[:,1].values
	testid = testdata.iloc[:,0].values

	scaler = StandardScaler()
	x_stand_train = scaler.fit_transform(x_train_raw)
	x_stand_test = scaler.transform(x_test) 
	

	rus = RUS(random_state=0)
	x_resampled, y_resampled = rus.fit_resample(x_stand_train, y_train_raw)


	full_train = np.column_stack((x_resampled, y_resampled))
	#print(full_train[0,:])
	# Outlier Removal
	# This is LocalOutlierFactor
	LOF = LocalOutlierFactor(n_neighbors=40, contamination=0.08)
	LOF.fit(full_train)
	y_pred_local = LOF.fit_predict(full_train)
	locations = np.where(y_pred_local == -1)

	# This is IsoForest Method
	rng = np.random.RandomState(42)
	IsoTree = IsolationForest(max_samples=100,random_state=rng,contamination=0.08)
	IsoTree.fit(full_train)
	y_pred_iso = IsoTree.predict(full_train)
	locations = np.where(y_pred_iso == -1)

	for i in range(len(y_pred_local)-1,-1,-1):
		if ((y_pred_iso[i] == -1) and (y_pred_local[i] == -1)):
			x_clean = np.delete(x_resampled, i, axis=0)
			y_clean = np.delete(y_resampled, i, axis=0)


	#sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
	sel1 = SelectFromModel(LinearSVC(class_weight="balanced"))
	sel1.fit(x_clean, y_clean)

	valid_select_svc = sel1.get_support()

	sel2 = SelectFromModel(RandomForestClassifier(n_estimators=100))
	sel2.fit(x_clean, y_clean)

	valid_select_rf = sel2.get_support()

	sel3 = SelectFromModel(GradientBoostingClassifier(n_estimators=100))
	sel3.fit(x_clean, y_clean)

	valid_select_gbc = sel3.get_support()

	valid_x = np.zeros(np.shape(x_clean)[1])
	for i in range(len(valid_select_rf)):
		if(valid_select_rf[i]==True and valid_select_svc[i]==True and valid_select_gbc[i]==True):
			valid_x[i] = True
		else:
			valid_x[i] = False
	x_invalid = np.where(valid_x == False)	
	print(len(x_invalid))

	train_selected_x = np.delete(x_clean,(x_invalid),axis=1)
	test_selected_x = np.delete(x_stand_test,(x_invalid),axis=1)

	x_train, x_val, y_train, y_val = train_test_split(train_selected_x, y_clean, test_size = 0.5)

	return x_train,y_train,x_val,y_val,test_selected_x,testid

# For testing preprocess purpose
#data_preprocess()







