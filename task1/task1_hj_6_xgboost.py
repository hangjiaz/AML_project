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
# read data file
x_traindata = pd.read_csv("X_train.csv", header=0)
y_traindata = pd.read_csv("y_train.csv", header=0)
x_train = x_traindata.iloc[:, 1:833].values
y_train = y_traindata.iloc[:, 1].values


x_testdata = pd.read_csv("X_test.csv", header=0)
x_test = x_testdata.iloc[:, 1:833].values
y_testid = x_testdata.iloc[:, 0].values

# mean imputation
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train_imputed = imp_mean.fit_transform(x_train)
x_test_imputed = imp_mean.fit_transform(x_test)
total_train = np.column_stack((x_train_imputed, y_train))


# data normalization
selector = VarianceThreshold(0.03)
x_train_imputed = selector.fit_transform(x_train_imputed)
x_test_imputed = selector.transform(x_test_imputed)
x_train_norm = dataNormalisation(x_train_imputed)
x_test_norm = dataNormalisation(x_test_imputed)

# pca transformation
cov = computeCov(x_train_norm)
[eigen_values,eigen_vectors] = computePCA(cov) 
  
x_train_reduced = transformData(eigen_vectors[:,0:8],x_train_norm)
x_test_reduced = transformData(eigen_vectors[:,0:8],x_test_norm)
    
    
# remove outliers based on Mahalanobis distance
transformed_data = transformData(eigen_vectors[:,0:2],x_train_norm )
x_train_reduced_rmo, y_train_rmo = remove_outlier(transformed_data,x_train_reduced, y_train) 
    
data_dmatrix = xgb.DMatrix(data = x_train_reduced_rmo, label = y_train_rmo )
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

#    
## poly feature
#polynomial_features= PolynomialFeatures(degree=2)
#x_poly = polynomial_features.fit_transform(x_train_reduced_rmo)
#x_test_poly = polynomial_features.transform(x_test_reduced)

# choose kernel using CV
x_train_for_cv = x_train_reduced_rmo
y_train_for_cv = y_train_rmo
paralst = []
result = 0 
kf = KFold(n_splits=10)

#for i in range(len(paralst)):
for train_index, test_index in kf.split(x_train_for_cv, y_train_for_cv):
    x_ktrain, y_ktrain = x_train_for_cv[train_index],y_train_for_cv[train_index]
    x_ktest, y_ktest = x_train_for_cv[test_index], y_train_for_cv[test_index]
    clf = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
    clf.fit(x_ktrain, y_ktrain)
    y_pred = clf.predict(x_ktest)
    result += r2_score(y_ktest, y_pred)
print(result/10)






# train model
#clf = LinearRegression()
#clf.fit(x_poly, y_train_rmo)
#y_pred = clf.predict(x_test_poly)
#
#with open('output.csv', 'w') as f:
#    f.write("{},{}\n".format("id", "y"))
#    for i in range(len(y_testid)):
#        f.write("{},{}\n".format(y_testid[i], y_pred[i]))
#



