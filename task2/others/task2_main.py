import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.covariance import MinCovDet as mcd
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
#from helper_function import *
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression,f_regression
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import GradientBoostingRegressor as GBR
from task2_preprocess import *
from model_zoo import *
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as BAC
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# def train_predict():
# 	x_resampled,y_resampled,x_test,testid = data_preprocess()
# 	# try svm 
# 	params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


# 	score = make_scorer(BAC)
# 	svm_model = GridSearchCV(SVC(), params_grid, cv=10, scoring=score)
# 	svm_model.fit(x_resampled, y_resampled)

# 	print('Best score for training data:', svm_model.best_score_,"\n") 

# 	# View the best parameters for the model found using grid search
# 	print('Best C:',svm_model.best_estimator_.C,"\n") 
# 	print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
# 	print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

# 	final_model = svm_model.best_estimator_
# 	y_pred = final_model.predict(x_test)
# 	return testid, y_pred


def write2file(y_testid, y_pred):
	with open('output.csv', 'w') as f:
		f.write("{},{}\n".format("id", "y"))
		for i in range(len(y_testid)):
			f.write("{},{}\n".format(y_testid[i], y_pred[i]))

	return

def main():
	x_resampled,y_resampled,x_test,testid = data_preprocess_1()
	#print(np.shape(x_resampled),np.shape(y_resampled),np.shape(x_val),np.shape(y_val))
	#y_pred, testid = mlp_2(x_resampled, y_resampled, x_val, y_val, x_test, testid)
	#y_pred, testid = light_gbm(x_resampled, y_resampled, x_val, y_val, x_test, testid)
	#model_zoo(x_resampled,y_resampled,x_val,y_val,x_test,testid)

	y_pred,testid = svc_model(x_resampled, y_resampled, x_test, testid)
	#y_pred,testid = linearsvc_model(x_resampled, y_resampled, x_val, y_val, x_test, testid)
	#y_pred,testid = mixture_model(x_resampled, y_resampled, x_val, y_val, x_test, testid)
	#y_pred,testid = xgb_model(x_resampled, y_resampled, x_val, y_val, x_test, testid)
	write2file(testid, y_pred)

if __name__ == "__main__":
	main()

def simply_true():
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

	return







