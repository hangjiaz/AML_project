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
from helper_function import *
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression,f_regression
from sklearn.decomposition import KernelPCA
from task1_preprocess_3 import *
import lightgbm as lgb


def train_predict():
	reduced_x_train, y_train, reduced_x_test, y_testid = data_preprocess()
	# create dataset for lightgbm
	x_train, x_val, y_train, y_val = train_test_split(reduced_x_train, y_train, test_size = 0.2)
	lgb_train = lgb.Dataset(x_train, y_train)
	lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

	# # specify the configurations as a dict
	params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    #'metric': {'l1', 'l2'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
	}

	print('Start training...')
	# train
	gbm = lgb.train(params,lgb_train, num_boost_round=500, feval=custom_r2, valid_sets={lgb_train, lgb_eval}, early_stopping_rounds=20)

	y_pred = gbm.predict(reduced_x_test, num_iteration=gbm.best_iteration)

	# clf = RidgeCV(alphas=[1e-3,1e-2,1e-1,1]).fit(x_train, y_train)
	# y_pred = clf.predict(reduced_x_test)
	# print("Score is ", clf.score(x_val,y_val))

	return y_testid, y_pred

def custom_r2(preds, train_data):
    labels = train_data.get_label()

    return 'r2', r2_score(labels, preds), True

def write2file(y_testid, y_pred):
	with open('output.csv', 'w') as f:
		f.write("{},{}\n".format("id", "y"))
		for i in range(len(y_testid)):
			f.write("{},{}\n".format(y_testid[i], y_pred[i]))

	return

testid, y_pred = train_predict()
write2file(testid, y_pred)








