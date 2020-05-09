import pandas as pd
import numpy as np
from sklearn.covariance import MinCovDet as mcd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as linalg
import pylab as pl
from scipy.spatial import distance


def computeCov(X=None):
    Xm = X - X.mean(axis=0)
    return 1.0/(Xm.shape[0]-1)*sp.dot(Xm.T,Xm)


def computePCA(matrix=None):
    #compute eigen values and vectors
    [eigen_values,eigen_vectors] = linalg.eig(matrix)
    #sort eigen vectors in decreasing order based on eigen values
    indices = sp.argsort(-eigen_values)
    return [sp.real(eigen_values[indices]), eigen_vectors[:,indices]]

'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(pcs=None,data=None):
    return sp.dot(pcs.T,data.T).T

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(evals=None):
    return evals/evals.sum()


'''
#################################################
  plot function
'''

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
'''
plot_color = ['#F7977A','#FDC68A','#A2D39C','#6ECFF6','#8493CA','#BC8DBF','#F6989D','#FFF79A']
def plotCumSumVariance(var=None,filename="cumsum.pdf"):
    pl.figure()
    pl.plot(sp.arange(var.shape[0]),sp.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    #Save file
    pl.savefig(filename)

'''
Plot Transformed Data
Input: transformed: data matrix (#sampels x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None,filename="exercise1.pdf"):
    pl.figure()
    ind_l = sp.unique(labels)
    legend = []
    for i,label in enumerate(ind_l):
        ind = sp.where(label==labels)[0]
        plot = pl.plot(transformed[ind,0],transformed[ind,1],'.',color=plot_color[i])
        legend.append(plot)
    pl.legend(ind_l,scatterpoints=1,numpoints=1,prop={'size':8},ncol=6,loc="upper right",fancybox=True)
    pl.xlabel("Transformed X Values")
    pl.ylabel("Transformed Y Values")
    pl.grid(True)
    #Save File
    pl.savefig(filename)


'''
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):
    Xm = X - X.mean(axis=0)
    return Xm/sp.std(Xm,axis=0)


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


def detectOutliers(data, std_num=2):
    dist = mcd(random_state=0)
    dist.fit(data)
    MDsqr = dist.mahalanobis(data)
    MD = np.sqrt(MDsqr)
    std = np.std(MD)
    k = std_num*std
    m = np.mean(MD)
    up_t = m + k
    low_t = m - k
    outliers = []
    for i in range(len(MD)):
        if (MD[i] >= up_t) or (MD[i] <= low_t):
            outliers = np.append(outliers, i)  # index of the outlier
    return outliers




def classify_y(y_train):
    y_class = np.zeros(y_train.shape[0])
    for i in range(y_train.shape[0]):
        if 40 < y_train[i] <= 50:
            y_class[i] = 45
        if 51 < y_train[i] <= 60:
            y_class[i] = 55
        if 61 < y_train[i] <= 70:
            y_class[i] = 65
        if 71 < y_train[i] <= 80:
            y_class[i] = 75
        if 81 < y_train[i] <= 90:
            y_class[i] = 85
        if 91 < y_train[i] <= 100:
            y_class[i] = 95
    return y_class

def remove_outlier(transformed_data, processed_data, processed_label, threshold):
    dist_lst = []
    outlier_idx = []
    for i in range(transformed_data.shape[0]):
       dist_lst.append(distance.euclidean(transformed_data[i][0], transformed_data[i][1])) 
       #delete outliers
       if dist_lst[i] > threshold:
           outlier_idx.append(i)
    for j in range(len(outlier_idx) - 1, -1, -1):
        processed_data= np.delete(processed_data, outlier_idx[j], axis=0)
        processed_label = np.delete(processed_label, outlier_idx[j], axis=0)
    return processed_data, processed_label
