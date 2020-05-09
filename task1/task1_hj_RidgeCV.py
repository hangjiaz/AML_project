import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.covariance import MinCovDet as mcd
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV, LassoCV, Ridge
from sklearn.metrics import r2_score

# read data file
x_traindata = pd.read_csv("X_train.csv", header=0)
y_traindata = pd.read_csv("y_train.csv", header=0)
x_train = x_traindata.iloc[:, 1:833].values
y_train = y_traindata.iloc[:, 1].values


x_testdata = pd.read_csv("X_test.csv", header=0)
x_test = x_testdata.iloc[:, 1:833].values
y_testid = x_testdata.iloc[:, 0].values

# svd imputation
x_train_imputed = svd_imputation(x_train)
x_test_imputed = svd_imputation(x_test)


# rescaling of data
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train_imputed)
x_test_scaled = scaler.transform(x_test_imputed)

# remove outliers based on Mahalanobis distance
index = detectOutliers(x_train_scaled)
for k in range(len(index) - 1, -1, -1):
    x_train_scaled_rmo= np.delete(x_train_scaled, index[k], axis=0)
    y_train_rmo = np.delete(y_train, index[k], axis=0)


# feature selection

 # determine variance threshod

#pars = [0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.004]
#result = np.zeros(len(pars))
#for i in range(len(pars)):
#    data =x_train_scaled.copy()
#    selector = VarianceThreshold(pars[i])
#    result[i]= selector.fit_transform(data).shape[1]
#plt.plot(result)

selector = VarianceThreshold(0.0008)
x_train_feature_selected = selector.fit_transform(x_train_scaled_rmo)
x_test_feature_selected = selector.transform(x_test_scaled )


clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,10])
clf.fit(x_train_feature_selected, y_train_rmo)
y_pred = clf.predict(x_test_feature_selected)

with open('output.csv', 'w') as f:
    f.write("{},{}\n".format("id", "y"))
    for i in range(len(y_testid)):
        f.write("{},{}\n".format(y_testid[i], y_pred[i]))



def mean_imputation(X=None):
    D_imputed = X.copy()
    #Impute each missing entry per feature with the mean of each feature
    for i in range(X.shape[1]):
        feature = X[:,i]
        #get indices for all non-nan values
        indices = np.where(~np.isnan(feature))[0]
        #compute mean for given feature
        mean = np.mean(feature[indices])
        #get nan indices
        nan_indices = np.where(np.isnan(feature))[0]
        #Update all nan values with the mean of each feature
        D_imputed[nan_indices,i] = mean
    return D_imputed



def svd_imputation(X=None,rank=None,tol=.1,max_iter=200):
    #get all nan indices
    nan_indices = np.where(np.isnan(X))
    #initialise all nan entries with the a mean imputation
    D_imputed = mean_imputation(X)
    #repeat approximation step until convergance
    for i in range(max_iter):
        D_old = D_imputed.copy()
        #SVD on mean_imputed data
        [L,d,R] = linalg.svd(D_imputed)
        d_mat = np.zeros((L.shape[0],R.shape[0]))
        d_mat[:d.shape[0],:d.shape[0]]=np.diag(d[:rank])
        #compute rank r approximation of D_imputed
        D_r = np.matrix(L[:,:rank])*d_mat*np.matrix(R[:rank,:])
        #update imputed entries according to the rank-r approximation
        imputed = D_r[nan_indices[0],nan_indices[1]]
        D_imputed[nan_indices[0],nan_indices[1]] = np.asarray(imputed)[0]
        #use Frobenius Norm to compute similarity between new matrix and the latter approximation
        fnorm = linalg.norm(D_old-D_imputed,ord="fro")
        if fnorm<tol:
            print("\t\t\t[SVD Imputation]: Converged after %d iterations"%(i+1))
            break
        if (i+1)>=max_iter:
            print("\t\t\t[SVD Imputation]: Maximum number of iterations reached (%d)"%(i+1))
    return D_imputed


def detectOutliers(data):
    dist = mcd(random_state=0)
    dist.fit(data)
    MDsqr = dist.mahalanobis(data)
    MD = np.sqrt(MDsqr)
    std = np.std(MD)
    k = 2*std
    m = np.mean(MD)
    up_t = m + k
    low_t = m - k
    outliers = []
    for i in range(len(MD)):
        if (MD[i] >= up_t) or (MD[i] <= low_t):
            outliers = np.append(outliers, i)  # index of the outlier
    return outliers

